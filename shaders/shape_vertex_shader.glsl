#version 410 core

layout (location = 0) in vec3 aPos;
layout (location = 2) in vec2 aTexCoord;

layout (location = 3) in vec4 aInstanceTransform0;
layout (location = 4) in vec4 aInstanceTransform1;
layout (location = 5) in vec4 aInstanceTransform2;
layout (location = 6) in vec4 aInstanceTransform3;

out vec2 v_uv;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    // Combine per‐instance transform with global model:
    mat4 instanceMatrix = mat4(
        aInstanceTransform0,
        aInstanceTransform1,
        aInstanceTransform2,
        aInstanceTransform3
    );
    mat4 worldMatrix = model * instanceMatrix;

    // Final position
    gl_Position = projection * view * (worldMatrix * vec4(aPos, 1.0));

    // Pass the UV coordinates
    v_uv = aTexCoord;
}
