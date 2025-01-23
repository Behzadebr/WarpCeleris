#version 410 core

in vec2 v_uv;
out vec4 FragColor;

// Uniform textures
uniform sampler2D waveTexture;   // texture unit #0
// uniform sampler2D bottomTexture; // texture unit #1

// Example uniforms
uniform float colorVal_min;
uniform float colorVal_max;

void main()
{
    // Just read wave.r, color from [0..1]
    float wave = texture(waveTexture, v_uv).r;

    // map wave into [0..1]:
    float t = (wave - colorVal_min) / (colorVal_max - colorVal_min);
    t = clamp(t, 0.0, 1.0);

    // pick a color from black->white
    vec3 color = vec3(t);

    FragColor = vec4(color, 1.0);
}
