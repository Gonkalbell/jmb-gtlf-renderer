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

@fragment
fn fs_skybox(vertex: SkyboxInterp) -> @location(0) vec4f {
    let origin = (res_camera.local_to_world * vec4f(0.0, 0.0, 0.0, 1.0)).xyz;

    var rq: ray_query;
    rayQueryInitialize(&rq, acc_struct, RayDesc(0u, 0xFFu, 0.1, 200.0, origin, vertex.ray_dir));
    rayQueryProceed(&rq);

    let intersection = rayQueryGetCommittedIntersection(&rq);
    var result =  vec4f(0.0);
    if intersection.kind == RAY_QUERY_INTERSECTION_NONE {
        result = textureSample(res_texture, res_sampler, vertex.ray_dir);
    } else {
        result = vec4f(intersection.barycentrics, 1.0 - intersection.barycentrics.x - intersection.barycentrics.y, 1.0);
    }
    return result;
}
