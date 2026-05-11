use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Quat, Vec3};
use std::ops::IndexMut;
use std::{borrow::Cow, iter, mem};
use wgpu::util::DeviceExt;

use crate::bind_groups::{self, AccStructure};
use crate::{DEPTH_FORMAT, shaders};

// from cube
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    _pos: [f32; 4],
    _tex_coord: [f32; 2],
}

fn vertex(pos: [i8; 3], tc: [i8; 2]) -> Vertex {
    Vertex {
        _pos: [pos[0] as f32, pos[1] as f32, pos[2] as f32, 1.0],
        _tex_coord: [tc[0] as f32, tc[1] as f32],
    }
}

fn create_vertices() -> (Vec<Vertex>, Vec<u16>) {
    let vertex_data = [
        // top (0, 0, 1)
        vertex([-1, -1, 1], [0, 0]),
        vertex([1, -1, 1], [1, 0]),
        vertex([1, 1, 1], [1, 1]),
        vertex([-1, 1, 1], [0, 1]),
        // bottom (0, 0, -1)
        vertex([-1, 1, -1], [1, 0]),
        vertex([1, 1, -1], [0, 0]),
        vertex([1, -1, -1], [0, 1]),
        vertex([-1, -1, -1], [1, 1]),
        // right (1, 0, 0)
        vertex([1, -1, -1], [0, 0]),
        vertex([1, 1, -1], [1, 0]),
        vertex([1, 1, 1], [1, 1]),
        vertex([1, -1, 1], [0, 1]),
        // left (-1, 0, 0)
        vertex([-1, -1, 1], [1, 0]),
        vertex([-1, 1, 1], [0, 0]),
        vertex([-1, 1, -1], [0, 1]),
        vertex([-1, -1, -1], [1, 1]),
        // front (0, 1, 0)
        vertex([1, 1, -1], [1, 0]),
        vertex([-1, 1, -1], [0, 0]),
        vertex([-1, 1, 1], [0, 1]),
        vertex([1, 1, 1], [1, 1]),
        // back (0, -1, 0)
        vertex([1, -1, 1], [0, 0]),
        vertex([-1, -1, 1], [1, 0]),
        vertex([-1, -1, -1], [1, 1]),
        vertex([1, -1, -1], [0, 1]),
    ];

    let index_data: &[u16] = &[
        0, 1, 2, 2, 3, 0, // top
        4, 5, 6, 6, 7, 4, // bottom
        8, 9, 10, 10, 11, 8, // right
        12, 13, 14, 14, 15, 12, // left
        16, 17, 18, 18, 19, 16, // front
        20, 21, 22, 22, 23, 20, // back
    ];

    (vertex_data.to_vec(), index_data.to_vec())
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Uniforms {
    view_inverse: Mat4,
    proj_inverse: Mat4,
}

#[derive(Clone, Debug)]
pub struct Example {
    pub bind_group: AccStructure,
}

impl Example {
    pub fn init(
        device: &wgpu::Device,
        queue: &wgpu::Queue
    ) -> Self {
        let side_count = 8;

        let (vertex_data, index_data) = create_vertices();

        let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertex_data),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::BLAS_INPUT,
        });

        let index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&index_data),
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::BLAS_INPUT,
        });

        let blas_geo_size_desc = wgpu::BlasTriangleGeometrySizeDescriptor {
            vertex_format: wgpu::VertexFormat::Float32x3,
            vertex_count: vertex_data.len() as u32,
            index_format: Some(wgpu::IndexFormat::Uint16),
            index_count: Some(index_data.len() as u32),
            flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
        };

        let blas = device.create_blas(
            &wgpu::CreateBlasDescriptor {
                label: None,
                flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                update_mode: wgpu::AccelerationStructureUpdateMode::Build,
            },
            wgpu::BlasGeometrySizeDescriptors::Triangles {
                descriptors: vec![blas_geo_size_desc.clone()],
            },
        );

        let tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
            label: None,
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
            max_instances: side_count * side_count,
        });

        let bind_group_layout = bind_groups::AccStructureLayout {
            acc_struct: &tlas,
        };

        let bind_group = bind_groups::AccStructure::from_bindings(device, bind_group_layout);

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        encoder.build_acceleration_structures(
            iter::once(&wgpu::BlasBuildEntry {
                blas: &blas,
                geometry: wgpu::BlasGeometries::TriangleGeometries(vec![
                    wgpu::BlasTriangleGeometry {
                        size: &blas_geo_size_desc,
                        vertex_buffer: &vertex_buf,
                        first_vertex: 0,
                        vertex_stride: mem::size_of::<Vertex>() as u64,
                        index_buffer: Some(&index_buf),
                        first_index: Some(0),
                        transform_buffer: None,
                        transform_buffer_offset: None,
                    },
                ]),
            }),
            iter::once(&tlas),
        );

        queue.submit(Some(encoder.finish()));

        Example {
            bind_group,
        }
    }
}
