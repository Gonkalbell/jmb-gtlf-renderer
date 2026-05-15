import super::bgroup_camera::res_camera;

enable wgpu_ray_query;

@group(1) @binding(0) var res_texture: texture_cube<f32>;
@group(1) @binding(1) var res_sampler: sampler;

@group(2) @binding(0) var acc_struct: acceleration_structure;

struct SkyboxInterp {
    @builtin(position) position: vec4f,
    @location(0) ray_dir: vec3<f32>,
};

@vertex
fn vs_skybox(@builtin(vertex_index) vertex_index: u32) -> SkyboxInterp {
    // hacky way to draw a large triangle
    let x = i32(vertex_index) / 2;
    let y = i32(vertex_index) & 1;
    let tc = vec2<f32>(
        f32(x) * 2.0,
        f32(y) * 2.0
    );
    let pos = vec4f(
        1.0 - 2.0 * tc,
        0.0, 1.0
    );

    var result: SkyboxInterp;
    result.position = pos;
    let dir = vec4f((res_camera.proj_to_local * pos).xyz, 0.0);
    result.ray_dir = (res_camera.local_to_world * dir).xyz;
    return result;
}

const TAU: f32 = 6.28318530718;
const PHI: f32 = 1.61803398875; // golden ratio

fn u32_to_color(n: u32) -> vec3f {
    let fi = f32(n);

    // Map n to an azimuthal angle using the golden ratio
    // Consecutive integers are separated by ~137.5°, the golden angle
    let theta = TAU * fract(fi / PHI);

    // Map n to a polar angle using a separate irrational stride
    // fract(n * phi_frac) gives a low-discrepancy sequence in [0, 1)
    let phi = acos(1.0 - 2.0 * fract(fi * 0.618033988749));

    // Convert spherical → Cartesian (unit sphere)
    let sp = sin(phi);
    let xyz = vec3f(
        sp * cos(theta),
        sp * sin(theta),
        cos(phi)
    );

    // Remap [-1, 1] → [0, 1] for use as RGB
    return xyz * 0.5 + 0.5;
}

fn u32x3_to_color(p: vec3<u32>) -> vec3f {
    let r = reverseBits(p.x);
    let g = reverseBits(p.y);
    let b = reverseBits(p.z);
    return vec3f(f32(r), f32(g), f32(b)) * (1.0 / f32(0xffffffffu));
}

@fragment
fn fs_skybox(vertex: SkyboxInterp) -> @location(0) vec4f {
    let origin = (res_camera.local_to_world * vec4f(0.0, 0.0, 0.0, 1.0)).xyz;

    var rq: ray_query;
    rayQueryInitialize(&rq, acc_struct, RayDesc(0u, 0xFFu, 0.01, 200.0, origin, vertex.ray_dir));
    rayQueryProceed(&rq);

    let intersection = rayQueryGetCommittedIntersection(&rq);
    var result =  vec4f(0.0);
    if intersection.kind == RAY_QUERY_INTERSECTION_NONE {
        result = textureSample(res_texture, res_sampler, vertex.ray_dir);
    } else {
        let w = 1.0 - intersection.barycentrics.x - intersection.barycentrics.y;
        let centrality = min(min(intersection.barycentrics.x, intersection.barycentrics.y), w) * 3.0;
        let input = vec3<u32>(
            intersection.geometry_index,
            intersection.primitive_index,
            intersection.instance_custom_data,
        );
        let color = u32x3_to_color(input);
        result = select(0.0, 1.0, centrality > 0.1) * vec4f(color, 1.0);
        // result = vec4f(intersection.barycentrics, 1.0 - intersection.barycentrics.x - intersection.barycentrics.y, 1.0);
    }
    return result;
}
