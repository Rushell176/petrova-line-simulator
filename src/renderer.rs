use std::path::Path;

use anyhow::{anyhow, Result};
use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;
use winit::window::Window;

use crate::geometry::{self, LineVertex, Particle, Vertex};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUbo {
    view_proj: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    cam_pos: [f32; 4],
    time: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PostUbo {
    time: [f32; 4],
    res: [f32; 4],
}

static HULL_ATTRS: [wgpu::VertexAttribute; 3] = [
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x3,
        offset: 0,
        shader_location: 0,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x3,
        offset: 12,
        shader_location: 1,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 24,
        shader_location: 2,
    },
];

static PARTICLE_ATTRS: [wgpu::VertexAttribute; 5] = [
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x3,
        offset: 0,
        shader_location: 0,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32,
        offset: 12,
        shader_location: 1,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32,
        offset: 16,
        shader_location: 2,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32,
        offset: 20,
        shader_location: 3,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x3,
        offset: 24,
        shader_location: 4,
    },
];

static LINE_ATTRS: [wgpu::VertexAttribute; 2] = [
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x3,
        offset: 0,
        shader_location: 0,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 12,
        shader_location: 1,
    },
];

fn hull_layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Vertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &HULL_ATTRS,
    }
}

fn particle_layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Particle>() as u64,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &PARTICLE_ATTRS,
    }
}

fn line_layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<LineVertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &LINE_ATTRS,
    }
}

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const MAX_LINE_VERTS: usize = 256;

pub struct Renderer {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: (u32, u32),

    _scene_tex: wgpu::Texture,
    scene_view: wgpu::TextureView,
    _depth_tex: wgpu::Texture,
    depth_view: wgpu::TextureView,

    post_bgl: wgpu::BindGroupLayout,
    post_bind_group: wgpu::BindGroup,
    post_pipeline: wgpu::RenderPipeline,

    sky_pipeline: wgpu::RenderPipeline,
    skybake_pipeline: wgpu::RenderPipeline,
    aurora_face_views: [wgpu::TextureView; 6],
    aurora_bg: wgpu::BindGroup,
    bake_face: u32,
    bake_tick: u32,
    hull_pipeline: wgpu::RenderPipeline,
    particle_pipeline: wgpu::RenderPipeline,
    line_pipeline: wgpu::RenderPipeline,

    sky_vb: wgpu::Buffer,
    sky_ib: wgpu::Buffer,
    sky_index_count: u32,
    hull_vb: wgpu::Buffer,
    hull_ib: wgpu::Buffer,
    hull_index_count: u32,
    particle_buf: wgpu::Buffer,
    particle_count: u32,
    tether_buf: wgpu::Buffer,

    camera_buf: wgpu::Buffer,
    camera_bg: wgpu::BindGroup,
    post_ubo: wgpu::Buffer,
    clear: wgpu::Color,
}

impl Renderer {
    pub fn new(window: std::sync::Arc<Window>) -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY | wgpu::Backends::GL,
            ..Default::default()
        });

        let surface = instance.create_surface(window.clone())?;

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .ok_or_else(|| anyhow!("no compatible GPU adapter found"))?;

        let info = adapter.get_info();
        println!("GPU: {} | backend: {:?}", info.name, info.backend);

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("petrova-device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None,
        ))?;

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);

        let size = {
            let s = window.inner_size();
            (s.width.max(1), s.height.max(1))
        };

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.0,
            height: size.1,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let (scene_tex, scene_view, depth_tex, depth_view) = create_targets(&device, size, format);

        let scene_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scene-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });
        let post_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("post-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("post.wgsl").into()),
        });

        let camera_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("camera-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let post_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("post-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let aurora_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("aurora-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::Cube,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let scene_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scene-layout"),
            bind_group_layouts: &[&camera_bgl, &aurora_bgl],
            push_constant_ranges: &[],
        });
        let bake_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bake-layout"),
            bind_group_layouts: &[&camera_bgl],
            push_constant_ranges: &[],
        });
        let post_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("post-layout"),
            bind_group_layouts: &[&post_bgl],
            push_constant_ranges: &[],
        });

        let initial_cam = CameraUbo {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            view: Mat4::IDENTITY.to_cols_array_2d(),
            proj: Mat4::IDENTITY.to_cols_array_2d(),
            cam_pos: [0.0, 0.0, 0.0, 1.0],
            time: [0.0, 0.0, 0.0, 0.0],
        };
        let camera_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera-ubo"),
            contents: bytemuck::bytes_of(&initial_cam),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let camera_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera-bg"),
            layout: &camera_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buf.as_entire_binding(),
            }],
        });

        let post_ubo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("post-ubo"),
            contents: bytemuck::bytes_of(&PostUbo {
                time: [0.0; 4],
                res: [size.0 as f32, size.1 as f32, 0.0, 0.0],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("linear-sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        let post_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("post-bg"),
            layout: &post_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&scene_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: post_ubo.as_entire_binding(),
                },
            ],
        });

        // The aurora is expensive to shade per pixel, so it lives in a small
        // cubemap baked a face at a time and is sampled cheaply by the sky.
        const AURORA_FACE: u32 = 512;
        let aurora_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("aurora-cube"),
            size: wgpu::Extent3d {
                width: AURORA_FACE,
                height: AURORA_FACE,
                depth_or_array_layers: 6,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let aurora_face_views: [wgpu::TextureView; 6] = std::array::from_fn(|i| {
            aurora_tex.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: i as u32,
                array_layer_count: Some(1),
                ..Default::default()
            })
        });
        let aurora_cube_view = aurora_tex.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        });
        let aurora_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("aurora-sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let aurora_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aurora-bg"),
            layout: &aurora_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&aurora_cube_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&aurora_sampler),
                },
            ],
        });

        let depth_state = wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        };
        let depth_state_no_write = wgpu::DepthStencilState {
            depth_write_enabled: false,
            ..depth_state.clone()
        };

        let additive = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        };

        let sky_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("sky-pipeline"),
            layout: Some(&scene_layout),
            vertex: wgpu::VertexState {
                module: &scene_shader,
                entry_point: "vs_sky",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[hull_layout()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &scene_shader,
                entry_point: "fs_sky",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(depth_state.clone()),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let skybake_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("skybake-pipeline"),
            layout: Some(&bake_layout),
            vertex: wgpu::VertexState {
                module: &scene_shader,
                entry_point: "vs_skybake",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &scene_shader,
                entry_point: "fs_skybake",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let hull_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("hull-pipeline"),
            layout: Some(&scene_layout),
            vertex: wgpu::VertexState {
                module: &scene_shader,
                entry_point: "vs_hull",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[hull_layout()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &scene_shader,
                entry_point: "fs_hull",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(depth_state),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let particle_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("particle-pipeline"),
            layout: Some(&scene_layout),
            vertex: wgpu::VertexState {
                module: &scene_shader,
                entry_point: "vs_particles",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[particle_layout()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &scene_shader,
                entry_point: "fs_particles",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(additive),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(depth_state_no_write.clone()),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let line_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("line-pipeline"),
            layout: Some(&scene_layout),
            vertex: wgpu::VertexState {
                module: &scene_shader,
                entry_point: "vs_line",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[line_layout()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &scene_shader,
                entry_point: "fs_line",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(depth_state_no_write),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let post_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("post-pipeline"),
            layout: Some(&post_layout),
            vertex: wgpu::VertexState {
                module: &post_shader,
                entry_point: "vs",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &post_shader,
                entry_point: "fs",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let (sky_verts, sky_indices) = geometry::build_sky();
        let sky_vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sky-vb"),
            contents: bytemuck::cast_slice(&sky_verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let sky_ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sky-ib"),
            contents: bytemuck::cast_slice(&sky_indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let (hull_verts, hull_indices) = geometry::build_ship();
        let hull_vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("hull-vb"),
            contents: bytemuck::cast_slice(&hull_verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let hull_ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("hull-ib"),
            contents: bytemuck::cast_slice(&hull_indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let particles = geometry::build_particles();
        let particle_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("particle-instances"),
            contents: bytemuck::cast_slice(&particles),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let tether_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tether"),
            size: (std::mem::size_of::<LineVertex>() * MAX_LINE_VERTS) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Bake all six aurora faces once so the first frames already show the
        // complete sky; afterwards one face is refreshed every other frame.
        {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("aurora-initial-bake"),
            });
            for face in 0..6u32 {
                encode_skybake(
                    &mut encoder,
                    &skybake_pipeline,
                    &camera_bg,
                    &aurora_face_views[face as usize],
                    face,
                );
            }
            queue.submit(Some(encoder.finish()));
        }

        Ok(Self {
            surface,
            device,
            queue,
            config,
            size,
            _scene_tex: scene_tex,
            scene_view,
            _depth_tex: depth_tex,
            depth_view,
            post_bgl,
            post_bind_group,
            post_pipeline,
            sky_pipeline,
            skybake_pipeline,
            aurora_face_views,
            aurora_bg,
            bake_face: 0,
            bake_tick: 0,
            hull_pipeline,
            particle_pipeline,
            line_pipeline,
            sky_vb,
            sky_ib,
            sky_index_count: sky_indices.len() as u32,
            hull_vb,
            hull_ib,
            hull_index_count: hull_indices.len() as u32,
            particle_buf,
            particle_count: particles.len() as u32,
            tether_buf,
            camera_buf,
            camera_bg,
            post_ubo,
            clear: wgpu::Color {
                r: 0.004,
                g: 0.001,
                b: 0.002,
                a: 1.0,
            },
        })
    }

    pub fn aspect(&self) -> f32 {
        self.size.0 as f32 / self.size.1.max(1) as f32
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        let width = width.max(1);
        let height = height.max(1);
        if (width, height) == self.size {
            return;
        }
        self.size = (width, height);
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);

        let (scene_tex, scene_view, depth_tex, depth_view) =
            create_targets(&self.device, self.size, self.config.format);
        self._scene_tex = scene_tex;
        self.scene_view = scene_view;
        self._depth_tex = depth_tex;
        self.depth_view = depth_view;

        let sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        self.post_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("post-bg"),
            layout: &self.post_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.scene_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.post_ubo.as_entire_binding(),
                },
            ],
        });
    }

    fn write_frame_uniforms(
        &self,
        view_proj: Mat4,
        view: Mat4,
        proj: Mat4,
        cam_pos: Vec3,
        time: f32,
        pj: f32,
        ps: f32,
        tether: &[LineVertex],
    ) -> usize {
        let cam = CameraUbo {
            view_proj: view_proj.to_cols_array_2d(),
            view: view.to_cols_array_2d(),
            proj: proj.to_cols_array_2d(),
            cam_pos: [cam_pos.x, cam_pos.y, cam_pos.z, 1.0],
            time: [time, pj, ps, 0.0],
        };
        self.queue
            .write_buffer(&self.camera_buf, 0, bytemuck::bytes_of(&cam));
        let post = PostUbo {
            time: [time, pj, ps, 0.0],
            res: [self.size.0 as f32, self.size.1 as f32, 0.0, 0.0],
        };
        self.queue
            .write_buffer(&self.post_ubo, 0, bytemuck::bytes_of(&post));

        let n = tether.len().min(MAX_LINE_VERTS);
        if n >= 2 {
            self.queue
                .write_buffer(&self.tether_buf, 0, bytemuck::cast_slice(&tether[..n]));
        }
        n
    }

    fn encode_scene(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        tether_len: usize,
        bake: Option<u32>,
        bake_on: bool,
    ) {
        if bake_on {
            match bake {
                Some(face) => encode_skybake(
                    encoder,
                    &self.skybake_pipeline,
                    &self.camera_bg,
                    &self.aurora_face_views[face as usize],
                    face,
                ),
                None => {
                    for face in 0..6u32 {
                        encode_skybake(
                            encoder,
                            &self.skybake_pipeline,
                            &self.camera_bg,
                            &self.aurora_face_views[face as usize],
                            face,
                        );
                    }
                }
            }
        }

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("scene-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.scene_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(self.clear),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        pass.set_bind_group(0, &self.camera_bg, &[]);
        pass.set_bind_group(1, &self.aurora_bg, &[]);

        pass.set_pipeline(&self.sky_pipeline);
        pass.set_vertex_buffer(0, self.sky_vb.slice(..));
        pass.set_index_buffer(self.sky_ib.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..self.sky_index_count, 0, 0..1);

        pass.set_pipeline(&self.hull_pipeline);
        pass.set_vertex_buffer(0, self.hull_vb.slice(..));
        pass.set_index_buffer(self.hull_ib.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..self.hull_index_count, 0, 0..1);

        if tether_len >= 2 {
            pass.set_pipeline(&self.line_pipeline);
            pass.set_vertex_buffer(0, self.tether_buf.slice(..));
            pass.draw(0..tether_len as u32, 0..1);
        }

        pass.set_pipeline(&self.particle_pipeline);
        pass.set_vertex_buffer(0, self.particle_buf.slice(..));
        pass.draw(0..6, 0..self.particle_count);
    }

    fn encode_post(&self, encoder: &mut wgpu::CommandEncoder, target: &wgpu::TextureView) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("post-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&self.post_pipeline);
        pass.set_bind_group(0, &self.post_bind_group, &[]);
        pass.draw(0..3, 0..1);
    }

    pub fn render(
        &mut self,
        view_proj: Mat4,
        view: Mat4,
        proj: Mat4,
        cam_pos: Vec3,
        time: f32,
        pj: f32,
        ps: f32,
        tether: &[LineVertex],
    ) -> Result<()> {
        let tether_len =
            self.write_frame_uniforms(view_proj, view, proj, cam_pos, time, pj, ps, tether);

        let frame = match self.surface.get_current_texture() {
            Ok(f) => f,
            Err(_) => {
                self.surface.configure(&self.device, &self.config);
                return Ok(());
            }
        };
        let target = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });
        self.bake_tick += 1;
        let bake_on = pj > 0.02;
        let bake = if self.bake_tick % 3 == 0 {
            let face = self.bake_face;
            self.bake_face = (self.bake_face + 1) % 6;
            Some(face)
        } else {
            None
        };
        self.encode_scene(&mut encoder, tether_len, bake, bake_on);
        self.encode_post(&mut encoder, &target);
        self.queue.submit(Some(encoder.finish()));
        frame.present();
        Ok(())
    }

    pub fn screenshot(
        &mut self,
        path: &Path,
        view_proj: Mat4,
        view: Mat4,
        proj: Mat4,
        cam_pos: Vec3,
        time: f32,
        pj: f32,
        ps: f32,
        tether: &[LineVertex],
    ) -> Result<()> {
        let tether_len =
            self.write_frame_uniforms(view_proj, view, proj, cam_pos, time, pj, ps, tether);
        let (w, h) = self.size;

        let shot_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shot"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.config.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let shot_view = shot_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("shot-frame"),
            });
        self.encode_scene(&mut encoder, tether_len, None, true);
        self.encode_post(&mut encoder, &shot_view);

        let bytes_per_pixel = 4u32;
        let unpadded = w * bytes_per_pixel;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded = unpadded.div_ceil(align) * align;
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (padded * h) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &shot_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &readback,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit(Some(encoder.finish()));

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()??;

        let data = slice.get_mapped_range();
        let bgra = matches!(
            self.config.format,
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
        );
        let mut rgba = vec![0u8; (w * h * 4) as usize];
        for y in 0..h as usize {
            let src = &data[y * padded as usize..y * padded as usize + unpadded as usize];
            let dst = &mut rgba[y * unpadded as usize..(y + 1) * unpadded as usize];
            if bgra {
                for x in 0..w as usize {
                    dst[x * 4 + 0] = src[x * 4 + 2];
                    dst[x * 4 + 1] = src[x * 4 + 1];
                    dst[x * 4 + 2] = src[x * 4 + 0];
                    dst[x * 4 + 3] = src[x * 4 + 3];
                }
            } else {
                dst.copy_from_slice(src);
            }
        }
        drop(data);
        readback.unmap();

        image::save_buffer(path, &rgba, w, h, image::ExtendedColorType::Rgba8)?;
        Ok(())
    }
}

fn encode_skybake(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::RenderPipeline,
    camera_bg: &wgpu::BindGroup,
    face_view: &wgpu::TextureView,
    face: u32,
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("aurora-bake-pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: face_view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, camera_bg, &[]);
    pass.draw(0..3, face..face + 1);
}

fn create_targets(
    device: &wgpu::Device,
    size: (u32, u32),
    format: wgpu::TextureFormat,
) -> (
    wgpu::Texture,
    wgpu::TextureView,
    wgpu::Texture,
    wgpu::TextureView,
) {
    let extent = wgpu::Extent3d {
        width: size.0,
        height: size.1,
        depth_or_array_layers: 1,
    };
    let scene_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("scene-color"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let scene_view = scene_tex.create_view(&wgpu::TextureViewDescriptor::default());
    let depth_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("scene-depth"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let depth_view = depth_tex.create_view(&wgpu::TextureViewDescriptor::default());
    (scene_tex, scene_view, depth_tex, depth_view)
}
