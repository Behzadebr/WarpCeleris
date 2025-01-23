# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import ctypes
import sys
import time
from collections import defaultdict
from typing import List, Optional, Tuple, Union
import os

import numpy as np

import warp as wp

Mat44 = Union[List[float], List[List[float]], np.ndarray]

wp.set_module_options({"enable_backward": False})

grid_vertex_shader = """
#version 330 core

uniform mat4 view;
uniform mat4 model;
uniform mat4 projection;

in vec3 position;

void main() {
    gl_Position = projection * view * model * vec4(position, 1.0);
}
"""

# Fragment shader source code
grid_fragment_shader = """
#version 330 core

out vec4 outColor;

void main() {
    outColor = vec4(0.5, 0.5, 0.5, 1.0);
}
"""

sky_vertex_shader = """
#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;

uniform mat4 view;
uniform mat4 inv_model;
uniform mat4 projection;
uniform vec3 viewPos;

out vec3 FragPos;
out vec2 TexCoord;

void main()
{
    vec4 worldPos = vec4(aPos + viewPos, 1.0);
    gl_Position = projection * view * inv_model * worldPos;

    FragPos = vec3(worldPos);
    TexCoord = aTexCoord;
}
"""

sky_fragment_shader = """
#version 330 core

out vec4 FragColor;

in vec3 FragPos;
in vec2 TexCoord;

uniform vec3 color1;
uniform vec3 color2;
uniform float farPlane;

uniform vec3 sunDirection;

void main()
{
    float y = tanh(FragPos.y/farPlane*10.0)*0.5+0.5;
    float height = sqrt(1.0-y);

    float s = pow(0.5, 1.0 / 10.0);
    s = 1.0 - clamp(s, 0.75, 1.0);

    vec3 haze = mix(vec3(1.0), color2 * 1.3, s);
    vec3 sky = mix(color1, haze, height / 1.3);

    float diff = max(dot(sunDirection, normalize(FragPos)), 0.0);
    vec3 sun = pow(diff, 32) * vec3(1.0, 0.8, 0.6) * 0.5;

    FragColor = vec4(sky + sun, 1.0);
}
"""

frame_vertex_shader = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;

out vec2 TexCoord;

void main() {
    gl_Position = vec4(aPos, 1.0);
    TexCoord = aTexCoord;
}
"""

frame_fragment_shader = """
#version 330 core
in vec2 TexCoord;

out vec4 FragColor;

uniform sampler2D textureSampler;

void main() {
    FragColor = texture(textureSampler, TexCoord);
}
"""

frame_depth_fragment_shader = """
#version 330 core
in vec2 TexCoord;

out vec4 FragColor;

uniform sampler2D textureSampler;

vec3 bourkeColorMap(float v) {
    vec3 c = vec3(1.0, 1.0, 1.0);

    v = clamp(v, 0.0, 1.0); // Ensures v is between 0 and 1

    if (v < 0.25) {
        c.r = 0.0;
        c.g = 4.0 * v;
    } else if (v < 0.5) {
        c.r = 0.0;
        c.b = 1.0 + 4.0 * (0.25 - v);
    } else if (v < 0.75) {
        c.r = 4.0 * (v - 0.5);
        c.b = 0.0;
    } else {
        c.g = 1.0 + 4.0 * (0.75 - v);
        c.b = 0.0;
    }

    return c;
}

void main() {
    float depth = texture(textureSampler, TexCoord).r;
    FragColor = vec4(bourkeColorMap(sqrt(1.0 - depth)), 1.0);
}
"""


@wp.kernel
def update_vbo_transforms(
    instance_id: wp.array(dtype=int),
    instance_body: wp.array(dtype=int),
    instance_transforms: wp.array(dtype=wp.transform),
    instance_scalings: wp.array(dtype=wp.vec3),
    body_q: wp.array(dtype=wp.transform),
    # outputs
    vbo_transforms: wp.array(dtype=wp.mat44),
):
    tid = wp.tid()
    i = instance_id[tid]
    X_ws = instance_transforms[i]
    if instance_body:
        body = instance_body[i]
        if body >= 0:
            if body_q:
                X_ws = body_q[body] * X_ws
            else:
                return
    p = wp.transform_get_translation(X_ws)
    q = wp.transform_get_rotation(X_ws)
    s = instance_scalings[i]
    rot = wp.quat_to_matrix(q)
    # transposed definition
    vbo_transforms[tid] = wp.mat44(
        rot[0, 0] * s[0],
        rot[1, 0] * s[0],
        rot[2, 0] * s[0],
        0.0,
        rot[0, 1] * s[1],
        rot[1, 1] * s[1],
        rot[2, 1] * s[1],
        0.0,
        rot[0, 2] * s[2],
        rot[1, 2] * s[2],
        rot[2, 2] * s[2],
        0.0,
        p[0],
        p[1],
        p[2],
        1.0,
    )


@wp.kernel
def update_vbo_vertices(
    points: wp.array(dtype=wp.vec3),
    scale: wp.vec3,
    # outputs
    vbo_vertices: wp.array(dtype=float, ndim=2),
):
    tid = wp.tid()
    p = points[tid]
    vbo_vertices[tid, 0] = p[0] * scale[0]
    vbo_vertices[tid, 1] = p[1] * scale[1]
    vbo_vertices[tid, 2] = p[2] * scale[2]


@wp.kernel
def update_points_positions(
    instance_positions: wp.array(dtype=wp.vec3),
    instance_scalings: wp.array(dtype=wp.vec3),
    # outputs
    vbo_transforms: wp.array(dtype=wp.mat44),
):
    tid = wp.tid()
    p = instance_positions[tid]
    s = wp.vec3(1.0)
    if instance_scalings:
        s = instance_scalings[tid]
    # transposed definition
    # fmt: off
    vbo_transforms[tid] = wp.mat44(
        s[0],  0.0,  0.0, 0.0,
         0.0, s[1],  0.0, 0.0,
         0.0,  0.0, s[2], 0.0,
        p[0], p[1], p[2], 1.0)
    # fmt: on


@wp.kernel
def update_line_transforms(
    lines: wp.array(dtype=wp.vec3, ndim=2),
    # outputs
    vbo_transforms: wp.array(dtype=wp.mat44),
):
    tid = wp.tid()
    p0 = lines[tid, 0]
    p1 = lines[tid, 1]
    p = 0.5 * (p0 + p1)
    d = p1 - p0
    s = wp.length(d)
    axis = wp.normalize(d)
    y_up = wp.vec3(0.0, 1.0, 0.0)
    angle = wp.acos(wp.dot(axis, y_up))
    axis = wp.normalize(wp.cross(axis, y_up))
    q = wp.quat_from_axis_angle(axis, -angle)
    rot = wp.quat_to_matrix(q)
    # transposed definition
    # fmt: off
    vbo_transforms[tid] = wp.mat44(
            rot[0, 0],     rot[1, 0],     rot[2, 0], 0.0,
        s * rot[0, 1], s * rot[1, 1], s * rot[2, 1], 0.0,
            rot[0, 2],     rot[1, 2],     rot[2, 2], 0.0,
                 p[0],          p[1],          p[2], 1.0,
    )


@wp.kernel
def copy_rgb_frame(
    input_img: wp.array(dtype=wp.uint8),
    width: int,
    height: int,
    # outputs
    output_img: wp.array(dtype=float, ndim=3),
):
    w, v = wp.tid()
    pixel = v * width + w
    pixel *= 3
    r = float(input_img[pixel + 0])
    g = float(input_img[pixel + 1])
    b = float(input_img[pixel + 2])
    # flip vertically (OpenGL coordinates start at bottom)
    v = height - v - 1
    output_img[v, w, 0] = r / 255.0
    output_img[v, w, 1] = g / 255.0
    output_img[v, w, 2] = b / 255.0


@wp.kernel
def copy_rgb_frame_uint8(
    input_img: wp.array(dtype=wp.uint8),
    width: int,
    height: int,
    # outputs
    output_img: wp.array(dtype=wp.uint8, ndim=3),
):
    w, v = wp.tid()
    pixel = v * width + w
    pixel *= 3
    # flip vertically (OpenGL coordinates start at bottom)
    v = height - v - 1
    output_img[v, w, 0] = input_img[pixel + 0]
    output_img[v, w, 1] = input_img[pixel + 1]
    output_img[v, w, 2] = input_img[pixel + 2]


@wp.kernel
def copy_depth_frame(
    input_img: wp.array(dtype=wp.float32),
    width: int,
    height: int,
    near: float,
    far: float,
    # outputs
    output_img: wp.array(dtype=wp.float32, ndim=3),
):
    w, v = wp.tid()
    pixel = v * width + w
    # flip vertically (OpenGL coordinates start at bottom)
    v = height - v - 1
    d = 2.0 * input_img[pixel] - 1.0
    d = 2.0 * near * far / ((far - near) * d - near - far)
    output_img[v, w, 0] = -d


@wp.kernel
def copy_rgb_frame_tiles(
    input_img: wp.array(dtype=wp.uint8),
    positions: wp.array(dtype=int, ndim=2),
    screen_width: int,
    screen_height: int,
    tile_height: int,
    # outputs
    output_img: wp.array(dtype=float, ndim=4),
):
    tile, x, y = wp.tid()
    p = positions[tile]
    qx = x + p[0]
    qy = y + p[1]
    pixel = qy * screen_width + qx
    # flip vertically (OpenGL coordinates start at bottom)
    y = tile_height - y - 1
    if qx >= screen_width or qy >= screen_height:
        output_img[tile, y, x, 0] = 0.0
        output_img[tile, y, x, 1] = 0.0
        output_img[tile, y, x, 2] = 0.0
        return  # prevent out-of-bounds access
    pixel *= 3
    r = float(input_img[pixel + 0])
    g = float(input_img[pixel + 1])
    b = float(input_img[pixel + 2])
    output_img[tile, y, x, 0] = r / 255.0
    output_img[tile, y, x, 1] = g / 255.0
    output_img[tile, y, x, 2] = b / 255.0


@wp.kernel
def copy_rgb_frame_tiles_uint8(
    input_img: wp.array(dtype=wp.uint8),
    positions: wp.array(dtype=int, ndim=2),
    screen_width: int,
    screen_height: int,
    tile_height: int,
    # outputs
    output_img: wp.array(dtype=wp.uint8, ndim=4),
):
    tile, x, y = wp.tid()
    p = positions[tile]
    qx = x + p[0]
    qy = y + p[1]
    pixel = qy * screen_width + qx
    # flip vertically (OpenGL coordinates start at bottom)
    y = tile_height - y - 1
    if qx >= screen_width or qy >= screen_height:
        output_img[tile, y, x, 0] = wp.uint8(0)
        output_img[tile, y, x, 1] = wp.uint8(0)
        output_img[tile, y, x, 2] = wp.uint8(0)
        return  # prevent out-of-bounds access
    pixel *= 3
    output_img[tile, y, x, 0] = input_img[pixel + 0]
    output_img[tile, y, x, 1] = input_img[pixel + 1]
    output_img[tile, y, x, 2] = input_img[pixel + 2]


@wp.kernel
def copy_depth_frame_tiles(
    input_img: wp.array(dtype=wp.float32),
    positions: wp.array(dtype=int, ndim=2),
    screen_width: int,
    screen_height: int,
    tile_height: int,
    near: float,
    far: float,
    # outputs
    output_img: wp.array(dtype=wp.float32, ndim=4),
):
    tile, x, y = wp.tid()
    p = positions[tile]
    qx = x + p[0]
    qy = y + p[1]
    pixel = qy * screen_width + qx
    # flip vertically (OpenGL coordinates start at bottom)
    y = tile_height - y - 1
    if qx >= screen_width or qy >= screen_height:
        output_img[tile, y, x, 0] = far
        return  # prevent out-of-bounds access
    d = 2.0 * input_img[pixel] - 1.0
    d = 2.0 * near * far / ((far - near) * d - near - far)
    output_img[tile, y, x, 0] = -d


@wp.kernel
def copy_rgb_frame_tile(
    input_img: wp.array(dtype=wp.uint8),
    offset_x: int,
    offset_y: int,
    screen_width: int,
    screen_height: int,
    tile_height: int,
    # outputs
    output_img: wp.array(dtype=float, ndim=4),
):
    tile, x, y = wp.tid()
    qx = x + offset_x
    qy = y + offset_y
    pixel = qy * screen_width + qx
    # flip vertically (OpenGL coordinates start at bottom)
    y = tile_height - y - 1
    if qx >= screen_width or qy >= screen_height:
        output_img[tile, y, x, 0] = 0.0
        output_img[tile, y, x, 1] = 0.0
        output_img[tile, y, x, 2] = 0.0
        return  # prevent out-of-bounds access
    pixel *= 3
    r = float(input_img[pixel + 0])
    g = float(input_img[pixel + 1])
    b = float(input_img[pixel + 2])
    output_img[tile, y, x, 0] = r / 255.0
    output_img[tile, y, x, 1] = g / 255.0
    output_img[tile, y, x, 2] = b / 255.0


@wp.kernel
def copy_rgb_frame_tile_uint8(
    input_img: wp.array(dtype=wp.uint8),
    offset_x: int,
    offset_y: int,
    screen_width: int,
    screen_height: int,
    tile_height: int,
    # outputs
    output_img: wp.array(dtype=wp.uint8, ndim=4),
):
    tile, x, y = wp.tid()
    qx = x + offset_x
    qy = y + offset_y
    pixel = qy * screen_width + qx
    # flip vertically (OpenGL coordinates start at bottom)
    y = tile_height - y - 1
    if qx >= screen_width or qy >= screen_height:
        output_img[tile, y, x, 0] = wp.uint8(0)
        output_img[tile, y, x, 1] = wp.uint8(0)
        output_img[tile, y, x, 2] = wp.uint8(0)
        return  # prevent out-of-bounds access
    pixel *= 3
    output_img[tile, y, x, 0] = input_img[pixel + 0]
    output_img[tile, y, x, 1] = input_img[pixel + 1]
    output_img[tile, y, x, 2] = input_img[pixel + 2]


def check_gl_error():
    from pyglet import gl

    error = gl.glGetError()
    if error != gl.GL_NO_ERROR:
        print(f"OpenGL error: {error}")


def str_buffer(string: str):
    return ctypes.c_char_p(string.encode("utf-8"))


def arr_pointer(arr: np.ndarray):
    return arr.astype(np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float))

##########################
def load_shader_source(file_name: str) -> str:
    shader_path = os.path.join(os.path.dirname(__file__), 'shaders', file_name)
    
    # Force Python to read the file as UTF-8:
    with open(shader_path, 'r', encoding='utf-8') as file:
        return file.read()
##########################


class OpenGLRenderer:
    """
    OpenGLRenderer is a simple OpenGL renderer for rendering 3D shapes and meshes.
    """

    # number of segments to use for rendering spheres, capsules, cones and cylinders
    default_num_segments = 32

    gl = None  # Class-level variable to hold the imported module

    @classmethod
    def initialize_gl(cls):
        if cls.gl is None:  # Only import if not already imported
            from pyglet import gl

            cls.gl = gl

    def __init__(
        self,
        sim_params=None,
        title="Warp",
        scaling=1.0,
        fps=60,
        up_axis="Y",
        screen_width=1024,
        screen_height=768,
        near_plane=1.0,
        far_plane=100.0,
        camera_fov=45.0,
        camera_pos=(0.0, 2.0, 10.0),
        camera_front=(0.0, 0.0, -1.0),
        camera_up=(0.0, 1.0, 0.0),
        background_color=(0.53, 0.8, 0.92),
        draw_grid=False,
        draw_sky=True,
        draw_axis=True,
        show_info=True,
        render_wireframe=False,
        render_depth=False,
        axis_scale=1.0,
        vsync=False,
        headless=None,
        enable_backface_culling=True,
        enable_mouse_interaction=True,
        enable_keyboard_interaction=True,
        device=None,
    ):
        """
        Args:

            title (str): The window title.
            scaling (float): The scaling factor for the scene.
            fps (int): The target frames per second.
            up_axis (str): The up axis of the scene. Can be "X", "Y", or "Z".
            screen_width (int): The width of the window.
            screen_height (int): The height of the window.
            near_plane (float): The near clipping plane.
            far_plane (float): The far clipping plane.
            camera_fov (float): The camera field of view in degrees.
            camera_pos (tuple): The initial camera position.
            camera_front (tuple): The initial camera front direction.
            camera_up (tuple): The initial camera up direction.
            background_color (tuple): The background color of the scene.
            draw_grid (bool): Whether to draw a grid indicating the ground plane.
            draw_sky (bool): Whether to draw a sky sphere.
            draw_axis (bool): Whether to draw the coordinate system axes.
            show_info (bool): Whether to overlay rendering information.
            render_wireframe (bool): Whether to render scene shapes as wireframes.
            render_depth (bool): Whether to show the depth buffer instead of the RGB image.
            axis_scale (float): The scale of the coordinate system axes being rendered (only if `draw_axis is True).
            vsync (bool): Whether to enable vertical synchronization.
            headless (bool): Whether to run in headless mode (no window is created). If None, the value is determined by the Pyglet configuration defined in `pyglet.options["headless"].
            enable_backface_culling (bool): Whether to enable backface culling.
            enable_mouse_interaction (bool): Whether to enable mouse interaction.
            enable_keyboard_interaction (bool): Whether to enable keyboard interaction.
            device (Devicelike): Where to store the internal data.

        Note:

            :class:OpenGLRenderer requires Pyglet (version >= 2.0, known to work on 2.0.7) to be installed.

            Headless rendering is supported via EGL on UNIX operating systems. To enable headless rendering, set the following pyglet options before importing `warp.render:

            .. code-block:: python

                import pyglet

                pyglet.options["headless"] = True

                import warp.render

                # OpenGLRenderer is instantiated with headless=True by default
                renderer = warp.render.OpenGLRenderer()
        """
        try:
            import pyglet

            # disable error checking for performance
            pyglet.options["debug_gl"] = False

            from pyglet.graphics.shader import Shader, ShaderProgram
            from pyglet.math import Vec3 as PyVec3

            OpenGLRenderer.initialize_gl()
            gl = OpenGLRenderer.gl
        except ImportError as e:
            raise Exception("OpenGLRenderer requires pyglet (version >= 2.0) to be installed.") from e

        self.camera_near_plane = near_plane
        self.camera_far_plane = far_plane
        self.camera_fov = camera_fov

        self.background_color = background_color
        self.draw_grid = draw_grid
        self.draw_sky = draw_sky
        self.draw_axis = draw_axis
        self.show_info = show_info
        self.render_wireframe = render_wireframe
        self.render_depth = render_depth
        self.enable_backface_culling = enable_backface_culling

        if device is None:
            self._device = wp.get_preferred_device()
        else:
            self._device = wp.get_device(device)

        self._title = title

        self.window = pyglet.window.Window(
            width=screen_width, height=screen_height, caption=title, resizable=True, vsync=vsync, visible=not headless
        )
        if headless is None:
            self.headless = pyglet.options.get("headless", False)
        else:
            self.headless = headless
        self.app = pyglet.app

        if not headless:
            # making window current opengl rendering context
            self.window.switch_to()

        self.screen_width, self.screen_height = self.window.get_framebuffer_size()

        self.enable_mouse_interaction = enable_mouse_interaction
        self.enable_keyboard_interaction = enable_keyboard_interaction

        self._camera_speed = 0.04
        if isinstance(up_axis, int):
            self._camera_axis = up_axis
        else:
            self._camera_axis = "XYZ".index(up_axis.upper())
        self._last_x, self._last_y = self.screen_width // 2, self.screen_height // 2
        self._first_mouse = True
        self._left_mouse_pressed = False
        self._keys_pressed = defaultdict(bool)
        self._input_processors = []
        self._key_callbacks = []

        self.render_2d_callbacks = []
        self.render_3d_callbacks = []

        self._camera_pos = PyVec3(0.0, 0.0, 0.0)
        self._camera_front = PyVec3(0.0, 0.0, -1.0)
        self._camera_up = PyVec3(0.0, 1.0, 0.0)
        self._scaling = scaling

        self._model_matrix = self.compute_model_matrix(self._camera_axis, scaling)
        self._inv_model_matrix = np.linalg.inv(self._model_matrix.reshape(4, 4)).flatten()
        self.update_view_matrix(cam_pos=camera_pos, cam_front=camera_front, cam_up=camera_up)
        self.update_projection_matrix()

        self._camera_front = self._camera_front.normalize()
        self._pitch = np.rad2deg(np.arcsin(self._camera_front.y))
        self._yaw = -np.rad2deg(np.arccos(self._camera_front.x / np.cos(np.deg2rad(self._pitch))))

        self._frame_dt = 1.0 / fps
        self.time = 0.0
        self._start_time = time.time()
        self.clock_time = 0.0
        self._paused = False
        self._frame_speed = 0.0
        self.skip_rendering = False
        self._skip_frame_counter = 0
        self._fps_update = 0.0
        self._fps_render = 0.0
        self._fps_alpha = 0.1  # low pass filter rate to update FPS stats

        self._body_name = {}
        self._shapes = []
        self._shape_geo_hash = {}
        self._shape_gl_buffers = {}
        self._shape_instances = defaultdict(list)
        self._instances = {}
        self._instance_custom_ids = {}
        self._instance_shape = {}
        self._instance_gl_buffers = {}
        self._instance_transform_gl_buffer = None
        self._instance_transform_cuda_buffer = None
        self._instance_color1_buffer = None
        self._instance_color2_buffer = None
        self._instance_count = 0
        self._wp_instance_ids = None
        self._wp_instance_custom_ids = None
        self._np_instance_visible = None
        self._instance_ids = None
        self._inverse_instance_ids = None
        self._wp_instance_transforms = None
        self._wp_instance_scalings = None
        self._wp_instance_bodies = None
        self._update_shape_instances = False
        self._add_shape_instances = False

        # additional shape instancer used for points and line rendering
        self._shape_instancers = {}

        # instancer for the arrow shapes sof the coordinate system axes
        self._axis_instancer = None

        # toggle tiled rendering
        self._tiled_rendering = False
        self._tile_instances = None
        self._tile_ncols = 0
        self._tile_nrows = 0
        self._tile_width = 0
        self._tile_height = 0
        self._tile_viewports = None
        self._tile_view_matrices = None
        self._tile_projection_matrices = None

        self._frame_texture = None
        self._frame_depth_texture = None
        self._frame_fbo = None
        self._frame_pbo = None

        if not headless:
            self.window.push_handlers(on_draw=self._draw)
            self.window.push_handlers(on_resize=self._window_resize_callback)
            self.window.push_handlers(on_key_press=self._key_press_callback)
            self.window.push_handlers(on_close=self._close_callback)

            self._key_handler = pyglet.window.key.KeyStateHandler()
            self.window.push_handlers(self._key_handler)

            self.window.on_mouse_scroll = self._scroll_callback
            self.window.on_mouse_drag = self._mouse_drag_callback

        gl.glClearColor(*self.background_color, 1)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthMask(True)
        gl.glDepthRange(0.0, 1.0)


        self._shape_shader = ShaderProgram(
            Shader(load_shader_source('shape_vertex_shader.glsl'), "vertex"),
            Shader(load_shader_source('shape_fragment_shader.glsl'), "fragment")
        )


        self._grid_shader = ShaderProgram(
            Shader(grid_vertex_shader, "vertex"), Shader(grid_fragment_shader, "fragment")
        )

        self._sun_direction = np.array((-0.2, 0.8, 0.3))
        self._sun_direction /= np.linalg.norm(self._sun_direction)
        with self._shape_shader:
            gl.glUniform3f(
                gl.glGetUniformLocation(self._shape_shader.id, str_buffer("sunDirection")), *self._sun_direction
            )
            gl.glUniform3f(gl.glGetUniformLocation(self._shape_shader.id, str_buffer("lightColor")), 1, 1, 1)
            self._loc_shape_model = gl.glGetUniformLocation(self._shape_shader.id, str_buffer("model"))
            self._loc_shape_view = gl.glGetUniformLocation(self._shape_shader.id, str_buffer("view"))
            self._loc_shape_projection = gl.glGetUniformLocation(self._shape_shader.id, str_buffer("projection"))
            self._loc_shape_view_pos = gl.glGetUniformLocation(self._shape_shader.id, str_buffer("viewPos"))
            gl.glUniform3f(self._loc_shape_view_pos, 0, 0, 10)

        with self._shape_shader:
            # waveTexture -> 0
            loc_wave = self.gl.glGetUniformLocation(self._shape_shader.id, str_buffer("waveTexture"))
            self.gl.glUniform1i(loc_wave, 0)

            # # bottomTexture -> 1
            # loc_bottom = self.gl.glGetUniformLocation(self._shape_shader.id, str_buffer("bottomTexture"))
            # self.gl.glUniform1i(loc_bottom, 1)

            # Fetch uniform locations for 'delta', 'base_depth', 'dx', 'dy', 'time', etc.:
            loc_delta      = self.gl.glGetUniformLocation(self._shape_shader.id, str_buffer("delta"))
            loc_base_depth = self.gl.glGetUniformLocation(self._shape_shader.id, str_buffer("base_depth"))
            loc_dx         = self.gl.glGetUniformLocation(self._shape_shader.id, str_buffer("dx"))
            loc_dy         = self.gl.glGetUniformLocation(self._shape_shader.id, str_buffer("dy"))
            loc_time       = self.gl.glGetUniformLocation(self._shape_shader.id, str_buffer("time"))

            # Domain size (used by the photorealistic shader)
            loc_width  = self.gl.glGetUniformLocation(self._shape_shader.id, str_buffer("WIDTH"))
            loc_height = self.gl.glGetUniformLocation(self._shape_shader.id, str_buffer("HEIGHT"))

            # Boundary types
            loc_west  = self.gl.glGetUniformLocation(self._shape_shader.id, str_buffer("west_boundary_type"))
            loc_east  = self.gl.glGetUniformLocation(self._shape_shader.id, str_buffer("east_boundary_type"))
            loc_south = self.gl.glGetUniformLocation(self._shape_shader.id, str_buffer("south_boundary_type"))
            loc_north = self.gl.glGetUniformLocation(self._shape_shader.id, str_buffer("north_boundary_type"))

            loc_colorValMin = self.gl.glGetUniformLocation(self._shape_shader.id, str_buffer("colorVal_min"))
            loc_colorValMax = self.gl.glGetUniformLocation(self._shape_shader.id, str_buffer("colorVal_max"))
            loc_cmapChoice = self.gl.glGetUniformLocation(self._shape_shader.id, str_buffer("colorMap_choice"))
            loc_surfToPlot = self.gl.glGetUniformLocation(self._shape_shader.id, str_buffer("surfaceToPlot"))
            loc_showBreak = self.gl.glGetUniformLocation(self._shape_shader.id, str_buffer("showBreaking"))
            loc_isOverlay = self.gl.glGetUniformLocation(self._shape_shader.id, str_buffer("IsOverlayMapLoaded"))

            loc_scaleX = self.gl.glGetUniformLocation(self._shape_shader.id, str_buffer("scaleX"))
            loc_scaleY = self.gl.glGetUniformLocation(self._shape_shader.id, str_buffer("scaleY"))
            loc_offsetX = self.gl.glGetUniformLocation(self._shape_shader.id, str_buffer("offsetX"))
            loc_offsetY = self.gl.glGetUniformLocation(self._shape_shader.id, str_buffer("offsetY"))


            self.gl.glUseProgram(self._shape_shader.id)

            # Assign values from sim_params (use 'delta' exactly)
            self.gl.glUniform1f(loc_delta,      float(sim_params.delta))
            self.gl.glUniform1f(loc_base_depth, float(sim_params.base_depth))
            self.gl.glUniform1f(loc_dx,         float(sim_params.dx))
            self.gl.glUniform1f(loc_dy,         float(sim_params.dy))
            self.gl.glUniform1f(loc_time,       0.0)

            self.gl.glUniform1i(loc_width,  sim_params.WIDTH)
            self.gl.glUniform1i(loc_height, sim_params.HEIGHT)

            self.gl.glUniform1i(loc_west,  int(sim_params.west_boundary_type))
            self.gl.glUniform1i(loc_east,  int(sim_params.east_boundary_type))
            self.gl.glUniform1i(loc_south, int(sim_params.south_boundary_type))
            self.gl.glUniform1i(loc_north, int(sim_params.north_boundary_type))

            self.gl.glUniform1f(loc_colorValMin, -1.0)
            self.gl.glUniform1f(loc_colorValMax,  1.0)
            self.gl.glUniform1i(loc_cmapChoice,   0)
            self.gl.glUniform1i(loc_surfToPlot,   0)
            self.gl.glUniform1i(loc_showBreak,    0)
            self.gl.glUniform1i(loc_isOverlay,    0)

            self.gl.glUniform1f(loc_scaleX, 1.0)
            self.gl.glUniform1f(loc_scaleY, 1.0)
            self.gl.glUniform1f(loc_offsetX, 0.0)
            self.gl.glUniform1f(loc_offsetY, 0.0)

            # Store the time location so we can update it each frame
            self._loc_shape_time = loc_time

        # BEGIN: Wave Texture Setup (CUDA-OpenGL Interop with PBO)

        self.wave_width = sim_params.WIDTH
        self.wave_height = sim_params.HEIGHT

        # 1) Create a Pixel Buffer Object (PBO) for the wave
        self.wave_pbo = self.gl.GLuint()
        self.gl.glGenBuffers(1, ctypes.byref(self.wave_pbo))
        self.gl.glBindBuffer(self.gl.GL_PIXEL_UNPACK_BUFFER, self.wave_pbo.value)

        # Allocate enough space for an RGB32F texture
        self.gl.glBufferData(
            self.gl.GL_PIXEL_UNPACK_BUFFER,
            self.wave_width * self.wave_height * 3 * ctypes.sizeof(ctypes.c_float),
            None,
            self.gl.GL_DYNAMIC_DRAW
        )
        self.gl.glBindBuffer(self.gl.GL_PIXEL_UNPACK_BUFFER, 0)

        # 2) Create and bind the OpenGL texture for wave heights
        self.wave_texture = self.gl.GLuint()
        self.gl.glGenTextures(1, ctypes.byref(self.wave_texture))
        self.gl.glBindTexture(self.gl.GL_TEXTURE_2D, self.wave_texture.value)

        self.gl.glTexImage2D(
            self.gl.GL_TEXTURE_2D,
            0,
            self.gl.GL_RGB32F,    # storing 3 floats
            self.wave_width,
            self.wave_height,
            0,
            self.gl.GL_RGB,
            self.gl.GL_FLOAT,
            None
        )

        # texture parameters
        self.gl.glTexParameteri(self.gl.GL_TEXTURE_2D, self.gl.GL_TEXTURE_WRAP_S, self.gl.GL_CLAMP_TO_EDGE)
        self.gl.glTexParameteri(self.gl.GL_TEXTURE_2D, self.gl.GL_TEXTURE_WRAP_T, self.gl.GL_CLAMP_TO_EDGE)
        self.gl.glTexParameteri(self.gl.GL_TEXTURE_2D, self.gl.GL_TEXTURE_MIN_FILTER, self.gl.GL_LINEAR)
        self.gl.glTexParameteri(self.gl.GL_TEXTURE_2D, self.gl.GL_TEXTURE_MAG_FILTER, self.gl.GL_LINEAR)
        self.gl.glBindTexture(self.gl.GL_TEXTURE_2D, 0)

        # 3) Register with Warp
        self.wave_cuda_buffer = wp.RegisteredGLBuffer(int(self.wave_pbo.value), self._device)

        # # === CREATE BOTTOM TEXTURE (STATIC) ===
        # self.bottom_texture = self.gl.GLuint()
        # self.gl.glGenTextures(1, ctypes.byref(self.bottom_texture))
        # self.gl.glBindTexture(self.gl.GL_TEXTURE_2D, self.bottom_texture.value)

        # For now, I use 1-channel float data (R32F), same width/height as wave for convenience:
        self.gl.glTexImage2D(
            self.gl.GL_TEXTURE_2D,
            0,
            self.gl.GL_R32F,  # 1 channel float
            self.wave_width,
            self.wave_height,
            0,
            self.gl.GL_RED,   # matching R32F
            self.gl.GL_FLOAT,
            None
        )

        # set texture parameters
        self.gl.glTexParameteri(self.gl.GL_TEXTURE_2D, self.gl.GL_TEXTURE_WRAP_S, self.gl.GL_CLAMP_TO_EDGE)
        self.gl.glTexParameteri(self.gl.GL_TEXTURE_2D, self.gl.GL_TEXTURE_WRAP_T, self.gl.GL_CLAMP_TO_EDGE)
        self.gl.glTexParameteri(self.gl.GL_TEXTURE_2D, self.gl.GL_TEXTURE_MIN_FILTER, self.gl.GL_LINEAR)
        self.gl.glTexParameteri(self.gl.GL_TEXTURE_2D, self.gl.GL_TEXTURE_MAG_FILTER, self.gl.GL_LINEAR)

        # unbind
        self.gl.glBindTexture(self.gl.GL_TEXTURE_2D, 0)

        # create grid data
        limit = 10.0
        ticks = np.linspace(-limit, limit, 21)
        grid_vertices = []
        for i in ticks:
            if self._camera_axis == 0:
                grid_vertices.extend([0, -limit, i, 0, limit, i])
                grid_vertices.extend([0, i, -limit, 0, i, limit])
            elif self._camera_axis == 1:
                grid_vertices.extend([-limit, 0, i, limit, 0, i])
                grid_vertices.extend([i, 0, -limit, i, 0, limit])
            elif self._camera_axis == 2:
                grid_vertices.extend([-limit, i, 0, limit, i, 0])
                grid_vertices.extend([i, -limit, 0, i, limit, 0])
        grid_vertices = np.array(grid_vertices, dtype=np.float32)
        self._grid_vertex_count = len(grid_vertices) // 3

        with self._grid_shader:
            self._grid_vao = gl.GLuint()
            gl.glGenVertexArrays(1, self._grid_vao)
            gl.glBindVertexArray(self._grid_vao)

            self._grid_vbo = gl.GLuint()
            gl.glGenBuffers(1, self._grid_vbo)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._grid_vbo)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, grid_vertices.nbytes, grid_vertices.ctypes.data, gl.GL_STATIC_DRAW)

            self._loc_grid_view = gl.glGetUniformLocation(self._grid_shader.id, str_buffer("view"))
            self._loc_grid_model = gl.glGetUniformLocation(self._grid_shader.id, str_buffer("model"))
            self._loc_grid_projection = gl.glGetUniformLocation(self._grid_shader.id, str_buffer("projection"))

            self._loc_grid_pos_attribute = gl.glGetAttribLocation(self._grid_shader.id, str_buffer("position"))
            gl.glVertexAttribPointer(self._loc_grid_pos_attribute, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
            gl.glEnableVertexAttribArray(self._loc_grid_pos_attribute)

        # create sky data
        self._sky_shader = ShaderProgram(Shader(sky_vertex_shader, "vertex"), Shader(sky_fragment_shader, "fragment"))

        with self._sky_shader:
            self._loc_sky_view = gl.glGetUniformLocation(self._sky_shader.id, str_buffer("view"))
            self._loc_sky_inv_model = gl.glGetUniformLocation(self._sky_shader.id, str_buffer("inv_model"))
            self._loc_sky_projection = gl.glGetUniformLocation(self._sky_shader.id, str_buffer("projection"))

            self._loc_sky_color1 = gl.glGetUniformLocation(self._sky_shader.id, str_buffer("color1"))
            self._loc_sky_color2 = gl.glGetUniformLocation(self._sky_shader.id, str_buffer("color2"))
            self._loc_sky_far_plane = gl.glGetUniformLocation(self._sky_shader.id, str_buffer("farPlane"))
            gl.glUniform3f(self._loc_sky_color1, *background_color)
            # glUniform3f(self._loc_sky_color2, *np.clip(np.array(background_color)+0.5, 0.0, 1.0))
            gl.glUniform3f(self._loc_sky_color2, 0.8, 0.4, 0.05)
            gl.glUniform1f(self._loc_sky_far_plane, self.camera_far_plane)
            self._loc_sky_view_pos = gl.glGetUniformLocation(self._sky_shader.id, str_buffer("viewPos"))
            gl.glUniform3f(
                gl.glGetUniformLocation(self._sky_shader.id, str_buffer("sunDirection")), *self._sun_direction
            )

        # create VAO, VBO, and EBO
        self._sky_vao = gl.GLuint()
        gl.glGenVertexArrays(1, self._sky_vao)
        gl.glBindVertexArray(self._sky_vao)

        vertices, indices = self._create_sphere_mesh(self.camera_far_plane * 0.9, 32, 32, reverse_winding=True)
        self._sky_tri_count = len(indices)

        self._sky_vbo = gl.GLuint()
        gl.glGenBuffers(1, self._sky_vbo)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._sky_vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices.ctypes.data, gl.GL_STATIC_DRAW)

        self._sky_ebo = gl.GLuint()
        gl.glGenBuffers(1, self._sky_ebo)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self._sky_ebo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices.ctypes.data, gl.GL_STATIC_DRAW)

        # set up vertex attributes
        vertex_stride = vertices.shape[1] * vertices.itemsize
        # positions
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, vertex_stride, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(0)
        # normals
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, vertex_stride, ctypes.c_void_p(3 * vertices.itemsize))
        gl.glEnableVertexAttribArray(1)
        # uv coordinates
        gl.glVertexAttribPointer(2, 2, gl.GL_FLOAT, gl.GL_FALSE, vertex_stride, ctypes.c_void_p(6 * vertices.itemsize))
        gl.glEnableVertexAttribArray(2)

        gl.glBindVertexArray(0)

        self._last_time = time.time()
        self._last_begin_frame_time = self._last_time
        self._last_end_frame_time = self._last_time

        # create frame buffer for rendering to a texture
        self._frame_texture = None
        self._frame_depth_texture = None
        self._frame_fbo = None
        self._setup_framebuffer()

        # fmt: off
        # set up VBO for the quad that is rendered to the user window with the texture
        self._frame_vertices = np.array([
            # Positions  TexCoords
            -1.0, -1.0,  0.0, 0.0,
             1.0, -1.0,  1.0, 0.0,
             1.0,  1.0,  1.0, 1.0,
            -1.0,  1.0,  0.0, 1.0
        ], dtype=np.float32)
        # fmt: on

        self._frame_indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)

        self._frame_vao = gl.GLuint()
        gl.glGenVertexArrays(1, self._frame_vao)
        gl.glBindVertexArray(self._frame_vao)

        self._frame_vbo = gl.GLuint()
        gl.glGenBuffers(1, self._frame_vbo)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._frame_vbo)
        gl.glBufferData(
            gl.GL_ARRAY_BUFFER, self._frame_vertices.nbytes, self._frame_vertices.ctypes.data, gl.GL_STATIC_DRAW
        )

        self._frame_ebo = gl.GLuint()
        gl.glGenBuffers(1, self._frame_ebo)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self._frame_ebo)
        gl.glBufferData(
            gl.GL_ELEMENT_ARRAY_BUFFER, self._frame_indices.nbytes, self._frame_indices.ctypes.data, gl.GL_STATIC_DRAW
        )

        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 4 * self._frame_vertices.itemsize, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(
            1, 2, gl.GL_FLOAT, gl.GL_FALSE, 4 * self._frame_vertices.itemsize, ctypes.c_void_p(2 * vertices.itemsize)
        )
        gl.glEnableVertexAttribArray(1)

        self._frame_shader = ShaderProgram(
            Shader(frame_vertex_shader, "vertex"), Shader(frame_fragment_shader, "fragment")
        )
        gl.glUseProgram(self._frame_shader.id)
        self._frame_loc_texture = gl.glGetUniformLocation(self._frame_shader.id, str_buffer("textureSampler"))

        self._frame_depth_shader = ShaderProgram(
            Shader(frame_vertex_shader, "vertex"), Shader(frame_depth_fragment_shader, "fragment")
        )
        gl.glUseProgram(self._frame_depth_shader.id)
        self._frame_loc_depth_texture = gl.glGetUniformLocation(
            self._frame_depth_shader.id, str_buffer("textureSampler")
        )

        # unbind the VBO and VAO
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindVertexArray(0)

        # update model matrix
        self.scaling = scaling

        check_gl_error()

        # create text to render stats on the screen
        self._info_label = pyglet.text.Label(
            "",
            font_name="Arial",
            font_size=12,
            color=(255, 255, 255, 255),
            x=10,
            y=10,
            anchor_x="left",
            anchor_y="top",
            multiline=True,
            width=400,
        )

        if not headless:
            # set up our own event handling so we can synchronously render frames
            # by calling update() in a loop
            from pyglet.window import Window

            Window._enable_event_queue = False

            self.window.switch_to()
            self.window.dispatch_pending_events()

            platform_event_loop = self.app.platform_event_loop
            platform_event_loop.start()

            # start event loop
            self.app.event_loop.dispatch_event("on_enter")


    @property
    def paused(self):
        return self._paused

    @paused.setter
    def paused(self, value):
        self._paused = value
        if value:
            self.window.set_caption(f"{self._title} (paused)")
        else:
            self.window.set_caption(self._title)

    @property
    def has_exit(self):
        return self.app.event_loop.has_exit

    def clear(self):
        gl = OpenGLRenderer.gl

        # Ensure all CUDA operations are complete before cleanup
        wp.synchronize()

        if not self.headless:
            self.app.event_loop.dispatch_event("on_exit")
            self.app.platform_event_loop.stop()

        if self._instance_transform_gl_buffer is not None:
            try:
                gl.glDeleteBuffers(1, self._instance_transform_gl_buffer)
                gl.glDeleteBuffers(1, self._instance_color1_buffer)
                gl.glDeleteBuffers(1, self._instance_color2_buffer)
            except gl.GLException:
                pass
        for vao, vbo, ebo, _, _vertex_cuda_buffer in self._shape_gl_buffers.values():
            try:
                gl.glDeleteVertexArrays(1, vao)
                gl.glDeleteBuffers(1, vbo)
                gl.glDeleteBuffers(1, ebo)
            except gl.GLException:
                pass

        # === Start: Delete Gradient Texture and PBO ===
        if hasattr(self, 'gradient_cuda_buffer'):
            del self.gradient_cuda_buffer  # Unregister the PBO first

        if hasattr(self, 'gradient_texture'):
            try:
                gl.glDeleteTextures(1, ctypes.byref(self.gradient_texture))  # Corrected
            except gl.GLException:
                pass

        if hasattr(self, 'gradient_pbo'):
            try:
                gl.glDeleteBuffers(1, ctypes.byref(self.gradient_pbo))  # Delete the PBO
            except gl.GLException:
                pass
        # === End: Delete Gradient Texture and PBO ===

        # Clear other internal data structures
        self._body_name.clear()
        self._shapes.clear()
        self._shape_geo_hash.clear()
        self._shape_gl_buffers.clear()
        self._shape_instances.clear()
        self._instances.clear()
        self._instance_shape.clear()
        self._instance_gl_buffers.clear()
        self._instance_transform_gl_buffer = None
        self._instance_transform_cuda_buffer = None
        self._instance_color1_buffer = None
        self._instance_color2_buffer = None
        self._wp_instance_ids = None
        self._wp_instance_custom_ids = None
        self._wp_instance_transforms = None
        self._wp_instance_scalings = None
        self._wp_instance_bodies = None
        self._np_instance_visible = None
        self._update_shape_instances = False




    def close(self):
        self.clear()
        self.window.close()

    @property
    def tiled_rendering(self):
        return self._tiled_rendering

    @tiled_rendering.setter
    def tiled_rendering(self, value):
        if value:
            assert self._tile_instances is not None, "Tiled rendering is not set up. Call setup_tiled_rendering first."
        self._tiled_rendering = value

    def setup_tiled_rendering(
        self,
        instances: List[List[int]],
        rescale_window: bool = False,
        tile_width: Optional[int] = None,
        tile_height: Optional[int] = None,
        tile_ncols: Optional[int] = None,
        tile_nrows: Optional[int] = None,
        tile_positions: Optional[List[Tuple[int]]] = None,
        tile_sizes: Optional[List[Tuple[int]]] = None,
        projection_matrices: Optional[List[Mat44]] = None,
        view_matrices: Optional[List[Mat44]] = None,
    ):
        """
        Set up tiled rendering where the render buffer is split into multiple tiles that can visualize
        different shape instances of the scene with different view and projection matrices.
        See :meth:get_pixels which allows to retrieve the pixels of for each tile.
        See :meth:update_tile which allows to update the shape instances, projection matrix, view matrix, tile size, or tile position for a given tile.

        :param instances: A list of lists of shape instance ids. Each list of shape instance ids
            will be rendered into a separate tile.
        :param rescale_window: If True, the window will be resized to fit the tiles.
        :param tile_width: The width of each tile in pixels (optional).
        :param tile_height: The height of each tile in pixels (optional).
        :param tile_ncols: The number of tiles rendered horizontally (optional). Will be considered
            if tile_width is set to compute the tile positions, unless tile_positions is defined.
        :param tile_positions: A list of (x, y) tuples specifying the position of each tile in pixels.
            If None, the tiles will be arranged in a square grid, or, if tile_ncols and tile_nrows
            is set, in a grid with the specified number of columns and rows.
        :param tile_sizes: A list of (width, height) tuples specifying the size of each tile in pixels.
            If None, the tiles will have the same size as specified by tile_width and tile_height.
        :param projection_matrices: A list of projection matrices for each tile (each view matrix is
            either a flattened 16-dimensional array or a 4x4 matrix).
            If the entire array is None, or only a view instances, the projection matrices for all, or these
            instances, respectively, will be derived from the current render settings.
        :param view_matrices: A list of view matrices for each tile (each view matrix is either a flattened
            16-dimensional array or a 4x4 matrix).
            If the entire array is None, or only a view instances, the view matrices for all, or these
            instances, respectively, will be derived from the current camera settings and be
            updated when the camera is moved.
        """

        assert len(instances) > 0 and all(isinstance(i, list) for i in instances), "Invalid tile instances."

        self._tile_instances = instances
        n = len(self._tile_instances)

        if tile_positions is None or tile_sizes is None:
            if tile_ncols is None or tile_nrows is None:
                # try to fit the tiles into a square
                self._tile_ncols = int(np.ceil(np.sqrt(n)))
                self._tile_nrows = int(np.ceil(n / float(self._tile_ncols)))
            else:
                self._tile_ncols = tile_ncols
                self._tile_nrows = tile_nrows
            self._tile_width = tile_width or max(32, self.screen_width // self._tile_ncols)
            self._tile_height = tile_height or max(32, self.screen_height // self._tile_nrows)
            self._tile_viewports = [
                (i * self._tile_width, j * self._tile_height, self._tile_width, self._tile_height)
                for i in range(self._tile_ncols)
                for j in range(self._tile_nrows)
            ]
            if rescale_window:
                self.window.set_size(self._tile_width * self._tile_ncols, self._tile_height * self._tile_nrows)
        else:
            assert (
                len(tile_positions) == n and len(tile_sizes) == n
            ), "Number of tiles does not match number of instances."
            self._tile_ncols = None
            self._tile_nrows = None
            self._tile_width = None
            self._tile_height = None
            if all(tile_sizes[i][0] == tile_sizes[0][0] for i in range(n)):
                # tiles all have the same width
                self._tile_width = tile_sizes[0][0]
            if all(tile_sizes[i][1] == tile_sizes[0][1] for i in range(n)):
                # tiles all have the same height
                self._tile_height = tile_sizes[0][1]
            self._tile_viewports = [(x, y, w, h) for (x, y), (w, h) in zip(tile_positions, tile_sizes)]

        if projection_matrices is None:
            projection_matrices = [None] * n
        self._tile_projection_matrices = []
        for i, p in enumerate(projection_matrices):
            if p is None:
                w, h = self._tile_viewports[i][2:]
                self._tile_projection_matrices.append(
                    self.compute_projection_matrix(
                        self.camera_fov, w / h, self.camera_near_plane, self.camera_far_plane
                    )
                )
            else:
                self._tile_projection_matrices.append(np.array(p).flatten())

        if view_matrices is None:
            self._tile_view_matrices = [None] * n
        else:
            self._tile_view_matrices = [np.array(m).flatten() for m in view_matrices]

        self._tiled_rendering = True

    def update_tile(
        self,
        tile_id,
        instances: Optional[List[int]] = None,
        projection_matrix: Optional[Mat44] = None,
        view_matrix: Optional[Mat44] = None,
        tile_size: Optional[Tuple[int]] = None,
        tile_position: Optional[Tuple[int]] = None,
    ):
        """
        Update the shape instances, projection matrix, view matrix, tile size, or tile position
        for a given tile given its index.

        :param tile_id: The index of the tile to update.
        :param instances: A list of shape instance ids (optional).
        :param projection_matrix: A projection matrix (optional).
        :param view_matrix: A view matrix (optional).
        :param tile_size: A (width, height) tuple specifying the size of the tile in pixels (optional).
        :param tile_position: A (x, y) tuple specifying the position of the tile in pixels (optional).
        """

        assert self._tile_instances is not None, "Tiled rendering is not set up. Call setup_tiled_rendering first."
        assert tile_id < len(self._tile_instances), "Invalid tile id."

        if instances is not None:
            self._tile_instances[tile_id] = instances
        if projection_matrix is not None:
            self._tile_projection_matrices[tile_id] = np.array(projection_matrix).flatten()
        if view_matrix is not None:
            self._tile_view_matrices[tile_id] = np.array(view_matrix).flatten()
        (x, y, w, h) = self._tile_viewports[tile_id]
        if tile_size is not None:
            w, h = tile_size
        if tile_position is not None:
            x, y = tile_position
        self._tile_viewports[tile_id] = (x, y, w, h)

    def _setup_framebuffer(self):
        gl = OpenGLRenderer.gl

        if self._frame_texture is None:
            self._frame_texture = gl.GLuint()
            gl.glGenTextures(1, self._frame_texture)
        if self._frame_depth_texture is None:
            self._frame_depth_texture = gl.GLuint()
            gl.glGenTextures(1, self._frame_depth_texture)

        # set up RGB texture
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._frame_texture)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGB,
            self.screen_width,
            self.screen_height,
            0,
            gl.GL_RGB,
            gl.GL_UNSIGNED_BYTE,
            None,
        )
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)

        # set up depth texture
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._frame_depth_texture)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_DEPTH_COMPONENT32,
            self.screen_width,
            self.screen_height,
            0,
            gl.GL_DEPTH_COMPONENT,
            gl.GL_FLOAT,
            None,
        )
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        # create a framebuffer object (FBO)
        if self._frame_fbo is None:
            self._frame_fbo = gl.GLuint()
            gl.glGenFramebuffers(1, self._frame_fbo)
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._frame_fbo)

            # attach the texture to the FBO as its color attachment
            gl.glFramebufferTexture2D(
                gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, self._frame_texture, 0
            )
            # attach the depth texture to the FBO as its depth attachment
            gl.glFramebufferTexture2D(
                gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_TEXTURE_2D, self._frame_depth_texture, 0
            )

            if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
                print("Framebuffer is not complete!", flush=True)
                gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
                sys.exit(1)

        # unbind the FBO (switch back to the default framebuffer)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

        if self._frame_pbo is None:
            self._frame_pbo = gl.GLuint()
            gl.glGenBuffers(1, self._frame_pbo)  # generate 1 buffer reference
        gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, self._frame_pbo)  # binding to this buffer

        # allocate memory for PBO
        rgb_bytes_per_pixel = 3
        depth_bytes_per_pixel = 4
        pixels = np.zeros(
            (self.screen_height, self.screen_width, rgb_bytes_per_pixel + depth_bytes_per_pixel), dtype=np.uint8
        )
        gl.glBufferData(gl.GL_PIXEL_PACK_BUFFER, pixels.nbytes, pixels.ctypes.data, gl.GL_DYNAMIC_DRAW)
        gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, 0)

    @staticmethod
    def compute_projection_matrix(
        fov: float,
        aspect_ratio: float,
        near_plane: float,
        far_plane: float,
    ) -> Mat44:
        """
        Compute a projection matrix given the field of view, aspect ratio, near plane, and far plane.

        :param fov: The field of view in degrees.
        :param aspect_ratio: The aspect ratio (width / height).
        :param near_plane: The near plane.
        :param far_plane: The far plane.
        :return: A projection matrix.
        """

        from pyglet.math import Mat4 as PyMat4

        return np.array(PyMat4.perspective_projection(aspect_ratio, near_plane, far_plane, fov))

    def update_projection_matrix(self):
        if self.screen_height == 0:
            return
        aspect_ratio = self.screen_width / self.screen_height
        self._projection_matrix = self.compute_projection_matrix(
            self.camera_fov, aspect_ratio, self.camera_near_plane, self.camera_far_plane
        )

    @property
    def camera_pos(self):
        return self._camera_pos

    @camera_pos.setter
    def camera_pos(self, value):
        self.update_view_matrix(cam_pos=value)

    @property
    def camera_front(self):
        return self._camera_front

    @camera_front.setter
    def camera_front(self, value):
        self.update_view_matrix(cam_front=value)

    @property
    def camera_up(self):
        return self._camera_up

    @camera_up.setter
    def camera_up(self, value):
        self.update_view_matrix(cam_up=value)

    def compute_view_matrix(self, cam_pos, cam_front, cam_up):
        from pyglet.math import Mat4, Vec3

        model = np.array(self._model_matrix).reshape((4, 4))
        cp = model @ np.array([*cam_pos / self._scaling, 1.0])
        cf = model @ np.array([*cam_front / self._scaling, 1.0])
        up = model @ np.array([*cam_up / self._scaling, 0.0])
        cp = Vec3(*cp[:3])
        cf = Vec3(*cf[:3])
        up = Vec3(*up[:3])
        return np.array(Mat4.look_at(cp, cp + cf, up), dtype=np.float32)

    def update_view_matrix(self, cam_pos=None, cam_front=None, cam_up=None, stiffness=1.0):
        from pyglet.math import Vec3

        if cam_pos is not None:
            self._camera_pos = self._camera_pos * (1.0 - stiffness) + Vec3(*cam_pos) * stiffness
        if cam_front is not None:
            self._camera_front = self._camera_front * (1.0 - stiffness) + Vec3(*cam_front) * stiffness
        if cam_up is not None:
            self._camera_up = self._camera_up * (1.0 - stiffness) + Vec3(*cam_up) * stiffness

        self._view_matrix = self.compute_view_matrix(self._camera_pos, self._camera_front, self._camera_up)

    @staticmethod
    def compute_model_matrix(camera_axis: int, scaling: float):
        if camera_axis == 0:
            return np.array((0, 0, scaling, 0, scaling, 0, 0, 0, 0, scaling, 0, 0, 0, 0, 0, 1), dtype=np.float32)
        elif camera_axis == 2:
            return np.array((0, scaling, 0, 0, 0, 0, scaling, 0, scaling, 0, 0, 0, 0, 0, 0, 1), dtype=np.float32)

        return np.array((scaling, 0, 0, 0, 0, scaling, 0, 0, 0, 0, scaling, 0, 0, 0, 0, 1), dtype=np.float32)

    def update_model_matrix(self, model_matrix: Optional[Mat44] = None):
        gl = OpenGLRenderer.gl

        if model_matrix is None:
            self._model_matrix = self.compute_model_matrix(self._camera_axis, self._scaling)
        else:
            self._model_matrix = np.array(model_matrix).flatten()
        self._inv_model_matrix = np.linalg.inv(self._model_matrix.reshape((4, 4))).flatten()
        # update model view matrix in shaders
        ptr = arr_pointer(self._model_matrix)
        gl.glUseProgram(self._shape_shader.id)
        gl.glUniformMatrix4fv(self._loc_shape_model, 1, gl.GL_FALSE, ptr)
        gl.glUseProgram(self._grid_shader.id)
        gl.glUniformMatrix4fv(self._loc_grid_model, 1, gl.GL_FALSE, ptr)
        # sky shader needs inverted model view matrix
        gl.glUseProgram(self._sky_shader.id)
        inv_ptr = arr_pointer(self._inv_model_matrix)
        gl.glUniformMatrix4fv(self._loc_sky_inv_model, 1, gl.GL_FALSE, inv_ptr)

    @property
    def num_tiles(self):
        return len(self._tile_instances)

    @property
    def tile_width(self):
        return self._tile_width

    @property
    def tile_height(self):
        return self._tile_height

    @property
    def num_shapes(self):
        return len(self._shapes)

    @property
    def num_instances(self):
        return self._instance_count

    @property
    def scaling(self):
        return self._scaling

    @scaling.setter
    def scaling(self, scaling):
        self._scaling = scaling
        self.update_model_matrix()

    def begin_frame(self, t: float = None):
        self._last_begin_frame_time = time.time()
        self.time = t or self.clock_time

    def end_frame(self):
        self._last_end_frame_time = time.time()
        if self._add_shape_instances:
            self.allocate_shape_instances()
        if self._update_shape_instances:
            self.update_shape_instances()
        self.update()
        while self.paused and self.is_running():
            self.update()

    def update(self):
        self.clock_time = time.time() - self._start_time
        update_duration = self.clock_time - self._last_time
        frame_duration = self._last_end_frame_time - self._last_begin_frame_time
        self._last_time = self.clock_time
        self._frame_speed = update_duration * 100.0

        if not self.headless:
            self.app.platform_event_loop.step(self._frame_dt * 1e-3)

        if not self.skip_rendering:
            self._skip_frame_counter += 1
            if self._skip_frame_counter > 100:
                self._skip_frame_counter = 0

            if frame_duration > 0.0:
                if self._fps_update is None:
                    self._fps_update = 1.0 / frame_duration
                else:
                    update = 1.0 / frame_duration
                    self._fps_update = (1.0 - self._fps_alpha) * self._fps_update + self._fps_alpha * update
            if update_duration > 0.0:
                if self._fps_render is None:
                    self._fps_render = 1.0 / update_duration
                else:
                    update = 1.0 / update_duration
                    self._fps_render = (1.0 - self._fps_alpha) * self._fps_render + self._fps_alpha * update

            if not self.headless:
                self.app.event_loop._redraw_windows(self._frame_dt * 1e-3)
            else:
                self._draw()

    def _draw(self):
        gl = OpenGLRenderer.gl

        if not self.headless:
            # catch key hold events
            self._process_inputs()

        if self.enable_backface_culling:
            gl.glEnable(gl.GL_CULL_FACE)
        else:
            gl.glDisable(gl.GL_CULL_FACE)

        if self._frame_fbo is not None:
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._frame_fbo)

        gl.glClearColor(*self.background_color, 1)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glBindVertexArray(0)

        if not self._tiled_rendering:
            if self.draw_grid:
                self._draw_grid()

            if self.draw_sky:
                self._draw_sky()

        view_mat_ptr = arr_pointer(self._view_matrix)
        projection_mat_ptr = arr_pointer(self._projection_matrix)
        gl.glUseProgram(self._shape_shader.id)
        gl.glUniformMatrix4fv(self._loc_shape_view, 1, gl.GL_FALSE, view_mat_ptr)
        gl.glUniform3f(self._loc_shape_view_pos, *self._camera_pos)
        gl.glUniformMatrix4fv(self._loc_shape_view, 1, gl.GL_FALSE, view_mat_ptr)
        gl.glUniformMatrix4fv(self._loc_shape_projection, 1, gl.GL_FALSE, projection_mat_ptr)

        if self.render_wireframe:
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)

        if self._tiled_rendering:
            self._render_scene_tiled()
        else:
            self._render_scene()

        for cb in self.render_3d_callbacks:
            cb()

        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)

        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glViewport(0, 0, self.screen_width, self.screen_height)

        # render frame buffer texture to screen
        if self._frame_fbo is not None:
            if self.render_depth:
                with self._frame_depth_shader:
                    gl.glActiveTexture(gl.GL_TEXTURE0)
                    gl.glBindTexture(gl.GL_TEXTURE_2D, self._frame_depth_texture)
                    gl.glUniform1i(self._frame_loc_depth_texture, 0)

                    gl.glBindVertexArray(self._frame_vao)
                    gl.glDrawElements(gl.GL_TRIANGLES, len(self._frame_indices), gl.GL_UNSIGNED_INT, None)
                    gl.glBindVertexArray(0)
                    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
            else:
                with self._frame_shader:
                    gl.glActiveTexture(gl.GL_TEXTURE0)
                    gl.glBindTexture(gl.GL_TEXTURE_2D, self._frame_texture)
                    gl.glUniform1i(self._frame_loc_texture, 0)

                    gl.glBindVertexArray(self._frame_vao)
                    gl.glDrawElements(gl.GL_TRIANGLES, len(self._frame_indices), gl.GL_UNSIGNED_INT, None)
                    gl.glBindVertexArray(0)
                    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        # check for OpenGL errors
        # check_gl_error()

        if self.show_info:
            gl.glClear(gl.GL_DEPTH_BUFFER_BIT)
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
            gl.glEnable(gl.GL_BLEND)

            text = f"""Sim Time: {self.time:.1f}
Update FPS: {self._fps_update:.1f}
Render FPS: {self._fps_render:.1f}
"""
            if self.paused:
                text += "\nPaused (press space to resume)"

            self._info_label.text = text
            self._info_label.y = self.screen_height - 5
            self._info_label.draw()

        for cb in self.render_2d_callbacks:
            cb()

    def _draw_grid(self, is_tiled=False):
        gl = OpenGLRenderer.gl

        if not is_tiled:
            gl.glUseProgram(self._grid_shader.id)

            gl.glUniformMatrix4fv(self._loc_grid_view, 1, gl.GL_FALSE, arr_pointer(self._view_matrix))
            gl.glUniformMatrix4fv(self._loc_grid_projection, 1, gl.GL_FALSE, arr_pointer(self._projection_matrix))

        gl.glBindVertexArray(self._grid_vao)
        gl.glDrawArrays(gl.GL_LINES, 0, self._grid_vertex_count)
        gl.glBindVertexArray(0)

    def _draw_sky(self, is_tiled=False):
        gl = OpenGLRenderer.gl

        if not is_tiled:
            gl.glUseProgram(self._sky_shader.id)

            gl.glUniformMatrix4fv(self._loc_sky_view, 1, gl.GL_FALSE, arr_pointer(self._view_matrix))
            gl.glUniformMatrix4fv(self._loc_sky_projection, 1, gl.GL_FALSE, arr_pointer(self._projection_matrix))
            gl.glUniform3f(self._loc_sky_view_pos, *self._camera_pos)

        gl.glBindVertexArray(self._sky_vao)
        gl.glDrawElements(gl.GL_TRIANGLES, self._sky_tri_count, gl.GL_UNSIGNED_INT, None)
        gl.glBindVertexArray(0)

    def _render_scene(self):
        gl = OpenGLRenderer.gl

        # Activate texture unit 0 and bind the wave texture
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.wave_texture)

        # # bottom => texture unit #1
        # gl.glActiveTexture(gl.GL_TEXTURE1)
        # gl.glBindTexture(gl.GL_TEXTURE_2D, self.bottom_texture)

        start_instance_idx = 0

        for shape, (vao, _, _, tri_count, _) in self._shape_gl_buffers.items():
            num_instances = len(self._shape_instances[shape])

            gl.glBindVertexArray(vao)
            gl.glDrawElementsInstancedBaseInstance(
                gl.GL_TRIANGLES, tri_count, gl.GL_UNSIGNED_INT, None, num_instances, start_instance_idx
            )

            start_instance_idx += num_instances

        # if self.draw_axis:
        #     self._axis_instancer.render()

        for instancer in self._shape_instancers.values():
            instancer.render()

        gl.glBindVertexArray(0)


    def _render_scene_tiled(self):
        gl = OpenGLRenderer.gl

        # Activate texture unit 0 and bind the wave texture
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.wave_texture)

        # # bottom => texture unit #1
        # gl.glActiveTexture(gl.GL_TEXTURE1)
        # gl.glBindTexture(gl.GL_TEXTURE_2D, self.bottom_texture)


        for i, viewport in enumerate(self._tile_viewports):
            projection_matrix_ptr = arr_pointer(self._tile_projection_matrices[i])
            view_matrix_ptr = arr_pointer(
                self._tile_view_matrices[i] if self._tile_view_matrices[i] is not None else self._view_matrix
            )

            gl.glViewport(*viewport)
            if self.draw_grid:
                gl.glUseProgram(self._grid_shader.id)
                gl.glUniformMatrix4fv(self._loc_grid_projection, 1, gl.GL_FALSE, projection_matrix_ptr)
                gl.glUniformMatrix4fv(self._loc_grid_view, 1, gl.GL_FALSE, view_matrix_ptr)
                self._draw_grid(is_tiled=True)

            if self.draw_sky:
                gl.glUseProgram(self._sky_shader.id)
                gl.glUniformMatrix4fv(self._loc_sky_projection, 1, gl.GL_FALSE, projection_matrix_ptr)
                gl.glUniformMatrix4fv(self._loc_sky_view, 1, gl.GL_FALSE, view_matrix_ptr)
                self._draw_sky(is_tiled=True)

            gl.glUseProgram(self._shape_shader.id)
            gl.glUniformMatrix4fv(self._loc_shape_projection, 1, gl.GL_FALSE, projection_matrix_ptr)
            gl.glUniformMatrix4fv(self._loc_shape_view, 1, gl.GL_FALSE, view_matrix_ptr)

            instances = self._tile_instances[i]

            for instance in instances:
                shape = self._instance_shape[instance]

                vao, _, _, tri_count, _ = self._shape_gl_buffers[shape]

                start_instance_idx = self._inverse_instance_ids[instance]

                gl.glBindVertexArray(vao)
                gl.glDrawElementsInstancedBaseInstance(
                    gl.GL_TRIANGLES, tri_count, gl.GL_UNSIGNED_INT, None, 1, start_instance_idx
                )

            # if self.draw_axis:
            #     self._axis_instancer.render()

            for instancer in self._shape_instancers.values():
                instancer.render()

        gl.glBindVertexArray(0)



    def _close_callback(self):
        self.close()

    def _mouse_drag_callback(self, x, y, dx, dy, buttons, modifiers):
        if not self.enable_mouse_interaction:
            return

        import pyglet

        if buttons & pyglet.window.mouse.LEFT:
            sensitivity = 0.1
            dx *= sensitivity
            dy *= sensitivity

            self._yaw += dx
            self._pitch += dy

            self._pitch = max(min(self._pitch, 89.0), -89.0)

            self._camera_front.x = np.cos(np.deg2rad(self._yaw)) * np.cos(np.deg2rad(self._pitch))
            self._camera_front.y = np.sin(np.deg2rad(self._pitch))
            self._camera_front.z = np.sin(np.deg2rad(self._yaw)) * np.cos(np.deg2rad(self._pitch))
            self._camera_front = self._camera_front.normalize()
            self.update_view_matrix()

    def _scroll_callback(self, x, y, scroll_x, scroll_y):
        if not self.enable_mouse_interaction:
            return

        self.camera_fov -= scroll_y
        self.camera_fov = max(min(self.camera_fov, 90.0), 15.0)
        self.update_projection_matrix()

    def _process_inputs(self):
        import pyglet
        from pyglet.math import Vec3 as PyVec3

        for cb in self._input_processors:
            if cb(self._key_handler) == pyglet.event.EVENT_HANDLED:
                return

        if self._key_handler[pyglet.window.key.W] or self._key_handler[pyglet.window.key.UP]:
            self._camera_pos += self._camera_front * (self._camera_speed * self._frame_speed)
            self.update_view_matrix()
        if self._key_handler[pyglet.window.key.S] or self._key_handler[pyglet.window.key.DOWN]:
            self._camera_pos -= self._camera_front * (self._camera_speed * self._frame_speed)
            self.update_view_matrix()
        if self._key_handler[pyglet.window.key.A] or self._key_handler[pyglet.window.key.LEFT]:
            camera_side = PyVec3.cross(self._camera_front, self._camera_up).normalize()
            self._camera_pos -= camera_side * (self._camera_speed * self._frame_speed)
            self.update_view_matrix()
        if self._key_handler[pyglet.window.key.D] or self._key_handler[pyglet.window.key.RIGHT]:
            camera_side = PyVec3.cross(self._camera_front, self._camera_up).normalize()
            self._camera_pos += camera_side * (self._camera_speed * self._frame_speed)
            self.update_view_matrix()

    def register_input_processor(self, callback):
        self._input_processors.append(callback)

    def _key_press_callback(self, symbol, modifiers):
        import pyglet

        if not self.enable_keyboard_interaction:
            return

        for cb in self._key_callbacks:
            if cb(symbol, modifiers) == pyglet.event.EVENT_HANDLED:
                return pyglet.event.EVENT_HANDLED

        if symbol == pyglet.window.key.ESCAPE:
            self.close()
        if symbol == pyglet.window.key.SPACE:
            self.paused = not self.paused
        if symbol == pyglet.window.key.TAB:
            self.skip_rendering = not self.skip_rendering
        if symbol == pyglet.window.key.C:
            self.draw_axis = not self.draw_axis
        if symbol == pyglet.window.key.G:
            self.draw_grid = not self.draw_grid
        if symbol == pyglet.window.key.I:
            self.show_info = not self.show_info
        if symbol == pyglet.window.key.X:
            self.render_wireframe = not self.render_wireframe
        if symbol == pyglet.window.key.T:
            self.render_depth = not self.render_depth
        if symbol == pyglet.window.key.B:
            self.enable_backface_culling = not self.enable_backface_culling

    def register_key_press_callback(self, callback):
        self._key_callbacks.append(callback)

    def _window_resize_callback(self, width, height):
        self._first_mouse = True
        self.screen_width, self.screen_height = self.window.get_framebuffer_size()
        self.update_projection_matrix()
        self._setup_framebuffer()

    def register_shape(self, geo_hash, vertices, indices, color1=None, color2=None):
        gl = OpenGLRenderer.gl

        shape = len(self._shapes)
        if color1 is None:
            color1 = tab10_color_map(len(self._shape_geo_hash))
        if color2 is None:
            color2 = np.clip(np.array(color1) + 0.25, 0.0, 1.0)
        # TODO check if we actually need to store the shape data
        self._shapes.append((vertices, indices, color1, color2, geo_hash))
        self._shape_geo_hash[geo_hash] = shape

        gl.glUseProgram(self._shape_shader.id)

        # Create VAO, VBO, and EBO
        vao = gl.GLuint()
        gl.glGenVertexArrays(1, vao)
        gl.glBindVertexArray(vao)

        vbo = gl.GLuint()
        gl.glGenBuffers(1, vbo)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices.ctypes.data, gl.GL_STATIC_DRAW)

        vertex_cuda_buffer = wp.RegisteredGLBuffer(int(vbo.value), self._device)

        ebo = gl.GLuint()
        gl.glGenBuffers(1, ebo)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ebo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices.ctypes.data, gl.GL_STATIC_DRAW)

        # Set up vertex attributes
        vertex_stride = vertices.shape[1] * vertices.itemsize
        # positions
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, vertex_stride, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(0)
        # normals
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, vertex_stride, ctypes.c_void_p(3 * vertices.itemsize))
        gl.glEnableVertexAttribArray(1)
        # uv coordinates
        gl.glVertexAttribPointer(2, 2, gl.GL_FLOAT, gl.GL_FALSE, vertex_stride, ctypes.c_void_p(6 * vertices.itemsize))
        gl.glEnableVertexAttribArray(2)

        gl.glBindVertexArray(0)

        self._shape_gl_buffers[shape] = (vao, vbo, ebo, len(indices), vertex_cuda_buffer)

        return shape

    def deregister_shape(self, shape):
        gl = OpenGLRenderer.gl

        if shape not in self._shape_gl_buffers:
            return

        vao, vbo, ebo, _, vertex_cuda_buffer = self._shape_gl_buffers[shape]
        try:
            gl.glDeleteVertexArrays(1, vao)
            gl.glDeleteBuffers(1, vbo)
            gl.glDeleteBuffers(1, ebo)
        except gl.GLException:
            pass

        _, _, _, _, geo_hash = self._shapes[shape]
        del self._shape_geo_hash[geo_hash]
        del self._shape_gl_buffers[shape]
        self._shapes.pop(shape)

    def add_shape_instance(
        self,
        name: str,
        shape: int,
        body,
        pos,
        rot,
        scale=(1.0, 1.0, 1.0),
        color1=None,
        color2=None,
        custom_index: int = -1,
        visible: bool = True,
    ):
        if color1 is None:
            color1 = self._shapes[shape][2]
        if color2 is None:
            color2 = self._shapes[shape][3]
        instance = len(self._instances)
        self._shape_instances[shape].append(instance)
        body = self._resolve_body_id(body)
        self._instances[name] = (instance, body, shape, [*pos, *rot], scale, color1, color2, visible)
        self._instance_shape[instance] = shape
        self._instance_custom_ids[instance] = custom_index
        self._add_shape_instances = True
        self._instance_count = len(self._instances)
        return instance

    def remove_shape_instance(self, name: str):
        if name not in self._instances:
            return

        instance, _, shape, _, _, _, _, _ = self._instances[name]

        self._shape_instances[shape].remove(instance)
        self._instance_count = len(self._instances)
        self._add_shape_instances = self._instance_count > 0
        del self._instance_shape[instance]
        del self._instance_custom_ids[instance]
        del self._instances[name]

    def update_instance_colors(self):
        gl = OpenGLRenderer.gl

        colors1, colors2 = [], []
        all_instances = list(self._instances.values())
        for _shape, instances in self._shape_instances.items():
            for i in instances:
                if i >= len(all_instances):
                    continue
                instance = all_instances[i]
                colors1.append(instance[5])
                colors2.append(instance[6])
        colors1 = np.array(colors1, dtype=np.float32)
        colors2 = np.array(colors2, dtype=np.float32)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._instance_color1_buffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, colors1.nbytes, colors1.ctypes.data, gl.GL_STATIC_DRAW)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._instance_color2_buffer)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, colors2.nbytes, colors2.ctypes.data, gl.GL_STATIC_DRAW)

    def allocate_shape_instances(self):
        gl = OpenGLRenderer.gl

        self._add_shape_instances = False
        self._wp_instance_transforms = wp.array(
            [instance[3] for instance in self._instances.values()], dtype=wp.transform, device=self._device
        )
        self._wp_instance_scalings = wp.array(
            [instance[4] for instance in self._instances.values()], dtype=wp.vec3, device=self._device
        )
        self._wp_instance_bodies = wp.array(
            [instance[1] for instance in self._instances.values()], dtype=wp.int32, device=self._device
        )

        gl.glUseProgram(self._shape_shader.id)
        if self._instance_transform_gl_buffer is not None:
            gl.glDeleteBuffers(1, self._instance_transform_gl_buffer)
            gl.glDeleteBuffers(1, self._instance_color1_buffer)
            gl.glDeleteBuffers(1, self._instance_color2_buffer)

        # create instance buffer and bind it as an instanced array
        self._instance_transform_gl_buffer = gl.GLuint()
        gl.glGenBuffers(1, self._instance_transform_gl_buffer)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._instance_transform_gl_buffer)

        transforms = np.tile(np.diag(np.ones(4, dtype=np.float32)), (len(self._instances), 1, 1))
        gl.glBufferData(gl.GL_ARRAY_BUFFER, transforms.nbytes, transforms.ctypes.data, gl.GL_DYNAMIC_DRAW)

        # create CUDA buffer for instance transforms
        self._instance_transform_cuda_buffer = wp.RegisteredGLBuffer(
            int(self._instance_transform_gl_buffer.value), self._device
        )

        # create color buffers
        self._instance_color1_buffer = gl.GLuint()
        gl.glGenBuffers(1, self._instance_color1_buffer)
        self._instance_color2_buffer = gl.GLuint()
        gl.glGenBuffers(1, self._instance_color2_buffer)

        self.update_instance_colors()

        # set up instance attribute pointers
        matrix_size = transforms[0].nbytes

        instance_ids = []
        instance_custom_ids = []
        instance_visible = []
        instances = list(self._instances.values())
        inverse_instance_ids = {}
        instance_count = 0
        colors_size = np.zeros(3, dtype=np.float32).nbytes
        for shape, (vao, _vbo, _ebo, _tri_count, _vertex_cuda_buffer) in self._shape_gl_buffers.items():
            gl.glBindVertexArray(vao)

            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._instance_transform_gl_buffer)

            # we can only send vec4s to the shader, so we need to split the instance transforms matrix into its column vectors
            for i in range(4):
                gl.glVertexAttribPointer(
                    3 + i, 4, gl.GL_FLOAT, gl.GL_FALSE, matrix_size, ctypes.c_void_p(i * matrix_size // 4)
                )
                gl.glEnableVertexAttribArray(3 + i)
                gl.glVertexAttribDivisor(3 + i, 1)

            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._instance_color1_buffer)
            gl.glVertexAttribPointer(7, 3, gl.GL_FLOAT, gl.GL_FALSE, colors_size, ctypes.c_void_p(0))
            gl.glEnableVertexAttribArray(7)
            gl.glVertexAttribDivisor(7, 1)

            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._instance_color2_buffer)
            gl.glVertexAttribPointer(8, 3, gl.GL_FLOAT, gl.GL_FALSE, colors_size, ctypes.c_void_p(0))
            gl.glEnableVertexAttribArray(8)
            gl.glVertexAttribDivisor(8, 1)

            instance_ids.extend(self._shape_instances[shape])
            for i in self._shape_instances[shape]:
                inverse_instance_ids[i] = instance_count
                instance_count += 1
                instance_custom_ids.append(self._instance_custom_ids[i])
                instance_visible.append(instances[i][7])

        # trigger update to the instance transforms
        self._update_shape_instances = True

        self._wp_instance_ids = wp.array(instance_ids, dtype=wp.int32, device=self._device)
        self._wp_instance_custom_ids = wp.array(instance_custom_ids, dtype=wp.int32, device=self._device)
        self._np_instance_visible = np.array(instance_visible)
        self._instance_ids = instance_ids
        self._inverse_instance_ids = inverse_instance_ids

        gl.glBindVertexArray(0)

    def update_shape_instance(self, name, pos=None, rot=None, color1=None, color2=None, visible=None):
        """Update the instance transform of the shape

        Args:
            name: The name of the shape
            pos: The position of the shape
            rot: The rotation of the shape
            color1: The first color of the checker pattern
            color2: The second color of the checker pattern
            visible: Whether the shape is visible
        """
        gl = OpenGLRenderer.gl

        if name in self._instances:
            i, body, shape, tf, scale, old_color1, old_color2, v = self._instances[name]
            if visible is None:
                visible = v
            new_tf = np.copy(tf)
            if pos is not None:
                new_tf[:3] = pos
            if rot is not None:
                new_tf[3:] = rot
            self._instances[name] = (
                i,
                body,
                shape,
                new_tf,
                scale,
                old_color1 if color1 is None else color1,
                old_color2 if color2 is None else color2,
                visible,
            )
            self._update_shape_instances = True
            if color1 is not None or color2 is not None:
                vao, vbo, ebo, tri_count, vertex_cuda_buffer = self._shape_gl_buffers[shape]
                gl.glBindVertexArray(vao)
                self.update_instance_colors()
                gl.glBindVertexArray(0)
            return True
        return False

    def update_shape_instances(self):
        with self._shape_shader:
            self._update_shape_instances = False
            self._wp_instance_transforms = wp.array(
                [instance[3] for instance in self._instances.values()], dtype=wp.transform, device=self._device
            )
            self.update_body_transforms(None)

    def update_body_transforms(self, body_tf: wp.array):
        if self._instance_transform_cuda_buffer is None:
            return

        body_q = None
        if body_tf is not None:
            if body_tf.device.is_cuda:
                body_q = body_tf
            else:
                body_q = body_tf.to(self._device)

        vbo_transforms = self._instance_transform_cuda_buffer.map(dtype=wp.mat44, shape=(self._instance_count,))

        wp.launch(
            update_vbo_transforms,
            dim=self._instance_count,
            inputs=[
                self._wp_instance_ids,
                self._wp_instance_bodies,
                self._wp_instance_transforms,
                self._wp_instance_scalings,
                body_q,
            ],
            outputs=[
                vbo_transforms,
            ],
            device=self._device,
        )

        self._instance_transform_cuda_buffer.unmap()

    def register_body(self, name):
        # register body name and return its ID
        if name not in self._body_name:
            self._body_name[name] = len(self._body_name)
        return self._body_name[name]

    def _resolve_body_id(self, body):
        if body is None:
            return -1
        if isinstance(body, int):
            return body
        return self._body_name[body]

    def is_running(self):
        return not self.app.event_loop.has_exit

    def save(self):
        # save just keeps the window open to allow the user to interact with the scene
        while not self.app.event_loop.has_exit:
            self.update()
        if self.app.event_loop.has_exit:
            self.clear()
            self.app.event_loop.exit()


    def render_plane(
        self,
        name: str,
        pos: tuple,
        rot: tuple,
        width: float,
        length: float,
        color: tuple = (1.0, 1.0, 1.0),
        color2=None,
        parent_body: str = None,
        is_template: bool = False,
        u_scaling=1.0,
        v_scaling=1.0,
    ):
        """Add a plane for visualization

        Args:
            name: The name of the plane
            pos: The position of the plane
            rot: The rotation of the plane
            width: The width of the plane
            length: The length of the plane
            color: The color of the plane
            texture: The texture of the plane (optional)
        """
        geo_hash = hash(("plane", width, length))
        if geo_hash in self._shape_geo_hash:
            shape = self._shape_geo_hash[geo_hash]
            if self.update_shape_instance(name, pos, rot):
                return shape
        else:
            faces = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)
            normal = (0.0, 1.0, 0.0)
            width = width if width > 0.0 else 100.0
            length = length if length > 0.0 else 100.0
            u = 1.0
            v = 1.0

            gfx_vertices = np.array([
                [-width,  0.0, -length,  *normal, 0.0, 0.0],
                [-width,  0.0,  length,  *normal, 0.0, 1.0],
                [ width,  0.0,  length,  *normal, 1.0, 1.0],
                [ width,  0.0, -length,  *normal, 1.0, 0.0],
            ], dtype=np.float32)

            shape = self.register_shape(geo_hash, gfx_vertices, faces, color1=color, color2=color2)
        if not is_template:
            body = self._resolve_body_id(parent_body)
            self.add_shape_instance(name, shape, body, pos, rot)
        return shape

    def render_ground(self, size: float = 1000.0, plane=None):
        """Add a ground plane for visualization

        Args:
            size: The size of the ground plane
        """
        color1 = (200 / 255, 200 / 255, 200 / 255)
        color2 = (150 / 255, 150 / 255, 150 / 255)
        sqh = np.sqrt(0.5)
        if self._camera_axis == 0:
            q = (0.0, 0.0, -sqh, sqh)
        elif self._camera_axis == 1:
            q = (0.0, 0.0, 0.0, 1.0)
        elif self._camera_axis == 2:
            q = (sqh, 0.0, 0.0, sqh)
        pos = (0.0, 0.0, 0.0)
        if plane is not None:
            normal = np.array(plane[:3])
            normal /= np.linalg.norm(normal)
            pos = plane[3] * normal
            if np.allclose(normal, (0.0, 1.0, 0.0)):
                # no rotation necessary
                q = (0.0, 0.0, 0.0, 1.0)
            else:
                c = np.cross(normal, (0.0, 1.0, 0.0))
                angle = np.arcsin(np.linalg.norm(c))
                axis = np.abs(c) / np.linalg.norm(c)
                q = wp.quat_from_axis_angle(axis, angle)
        return self.render_plane(
            "ground",
            pos,
            q,
            size,
            size,
            color1,
            color2=color2,
            u_scaling=1.0,
            v_scaling=1.0,
        )


    def render_ref(self, name: str, path: str, pos: tuple, rot: tuple, scale: tuple, color: tuple = None):
        """
        Create a reference (instance) with the given name to the given path.
        """

        if path in self._instances:
            _, body, shape, _, original_scale, color1, color2 = self._instances[path]
            if color is not None:
                color1 = color2 = color
            self.add_shape_instance(name, shape, body, pos, rot, scale or original_scale, color1, color2)
            return

        raise Exception("Cannot create reference to path: " + path)

    @staticmethod
    def _create_sphere_mesh(
        radius=1.0,
        num_latitudes=default_num_segments,
        num_longitudes=default_num_segments,
        reverse_winding=False,
    ):
        vertices = []
        indices = []

        for i in range(num_latitudes + 1):
            theta = i * np.pi / num_latitudes
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)

            for j in range(num_longitudes + 1):
                phi = j * 2 * np.pi / num_longitudes
                sin_phi = np.sin(phi)
                cos_phi = np.cos(phi)

                x = cos_phi * sin_theta
                y = cos_theta
                z = sin_phi * sin_theta

                u = float(j) / num_longitudes
                v = float(i) / num_latitudes

                vertices.append([x * radius, y * radius, z * radius, x, y, z, u, v])

        for i in range(num_latitudes):
            for j in range(num_longitudes):
                first = i * (num_longitudes + 1) + j
                second = first + num_longitudes + 1

                if reverse_winding:
                    indices.extend([first, second, first + 1, second, second + 1, first + 1])
                else:
                    indices.extend([first, first + 1, second, second, first + 1, second + 1])

        return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32)

    def set_time(self, t: float):
        self.gl.glUseProgram(self._shape_shader.id)
        self.gl.glUniform1f(self._loc_shape_time, t)


    def update_wave_texture(self, wave_array: wp.array, width: int, height: int):
        """
        Copy wave_array (wp.vec4, shape=(width, height)) -> wave_texture (RGB32F).
        Because we are mapping the PBO as (height, width), the kernel writes out_tex[y,x].
        """
        from kernels.wave_texture_kernels import populate_wave_texture

        # If user changed sim domain size mid-run, re-allocate wave_pbo + wave_texture
        if width != self.wave_width or height != self.wave_height:
            self.wave_width  = width
            self.wave_height = height
            # reallocate PBO
            self.gl.glBindBuffer(self.gl.GL_PIXEL_UNPACK_BUFFER, self.wave_pbo.value)
            self.gl.glBufferData(
                self.gl.GL_PIXEL_UNPACK_BUFFER,
                self.wave_width * self.wave_height * 3 * ctypes.sizeof(ctypes.c_float),  # RGB32F
                None,
                self.gl.GL_DYNAMIC_DRAW
            )
            self.gl.glBindBuffer(self.gl.GL_PIXEL_UNPACK_BUFFER, 0)

            # reallocate texture
            self.gl.glBindTexture(self.gl.GL_TEXTURE_2D, self.wave_texture.value)
            self.gl.glTexImage2D(
                self.gl.GL_TEXTURE_2D,
                0,
                self.gl.GL_RGB32F,
                self.wave_width,
                self.wave_height,
                0,
                self.gl.GL_RGB,
                self.gl.GL_FLOAT,
                None
            )
            self.gl.glBindTexture(self.gl.GL_TEXTURE_2D, 0)

        # 1) Map the GL PBO as a Warp array with shape=(height, width)
        wave_buffer = self.wave_cuda_buffer.map(dtype=wp.vec3,
                                                shape=(self.wave_height, self.wave_width))

        # 2) Launch kernel
        wp.launch(
            populate_wave_texture,
            dim=(width, height),
            inputs=[width, height, wave_array],
            outputs=[wave_buffer],
            device=self._device
        )

        # 3) Unmap the GL buffer by calling wave_cuda_buffer.unmap()
        self.wave_cuda_buffer.unmap()

        # 4) Push data into the actual GL texture
        self.gl.glBindBuffer(self.gl.GL_PIXEL_UNPACK_BUFFER, self.wave_pbo.value)
        self.gl.glBindTexture(self.gl.GL_TEXTURE_2D, self.wave_texture.value)

        self.gl.glTexSubImage2D(
            self.gl.GL_TEXTURE_2D,
            0,
            0,
            0,
            self.wave_width,
            self.wave_height,
            self.gl.GL_RGB,
            self.gl.GL_FLOAT,
            ctypes.c_void_p(0)
        )

        self.gl.glBindBuffer(self.gl.GL_PIXEL_UNPACK_BUFFER, 0)
        self.gl.glBindTexture(self.gl.GL_TEXTURE_2D, 0)

    

if __name__ == "__main__":
    renderer = OpenGLRenderer()