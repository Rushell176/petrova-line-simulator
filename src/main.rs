mod geometry;
mod renderer;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use glam::{Mat4, Vec3};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{DeviceEvent, DeviceId, ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Fullscreen, Window, WindowId};

use geometry::LineVertex;
use renderer::Renderer;

const EYE_HEIGHT_OFFSET: f32 = 0.35;

struct Keys {
    w: bool,
    a: bool,
    s: bool,
    d: bool,
    up: bool,
    down: bool,
}

impl Default for Keys {
    fn default() -> Self {
        Self {
            w: false,
            a: false,
            s: false,
            d: false,
            up: false,
            down: false,
        }
    }
}

struct Camera {
    pos: Vec3,
    yaw: f32,
    pitch: f32,
    vel: Vec3,
    anchor: Vec3,
    limit: f32,
}

impl Camera {
    fn new(pos: Vec3, yaw_deg: f32, pitch_deg: f32) -> Self {
        Self {
            pos,
            yaw: yaw_deg.to_radians(),
            pitch: pitch_deg.to_radians(),
            vel: Vec3::ZERO,
            anchor: Vec3::new(0.0, 3.42, -1.0),
            limit: 7.5,
        }
    }

    fn forward(&self) -> Vec3 {
        let cp = self.pitch.cos();
        Vec3::new(self.yaw.sin() * cp, self.pitch.sin(), self.yaw.cos() * cp)
    }

    fn matrices(&self, aspect: f32) -> (Mat4, Mat4, Mat4) {
        let view = Mat4::look_to_rh(self.pos, self.forward(), Vec3::Y);
        let proj = Mat4::perspective_rh(65f32.to_radians(), aspect.max(0.1), 0.05, 500.0);
        (proj * view, view, proj)
    }

    fn update(&mut self, dt: f32, keys: &Keys, mouse_dx: f32, mouse_dy: f32) {
        self.yaw -= mouse_dx * 0.0022;
        self.pitch = (self.pitch - mouse_dy * 0.0022).clamp(-1.5, 1.5);

        let fwd = self.forward();
        let right = fwd.cross(Vec3::Y).normalize_or_zero();
        let mut acc = Vec3::ZERO;
        if keys.w {
            acc += fwd;
        }
        if keys.s {
            acc -= fwd;
        }
        if keys.d {
            acc += right;
        }
        if keys.a {
            acc -= right;
        }
        if keys.up {
            acc += Vec3::Y;
        }
        if keys.down {
            acc -= Vec3::Y;
        }
        if acc.length_squared() > 0.0 {
            self.vel += acc.normalize() * 7.0 * dt;
        }
        self.vel *= (-1.1 * dt).exp();

        let speed = self.vel.length();
        if speed > 6.0 {
            self.vel *= 6.0 / speed;
        }
        self.pos += self.vel * dt;

        let rel = self.pos - self.anchor;
        let dist = rel.length();
        if dist > self.limit {
            let n = rel / dist;
            self.pos = self.anchor + n * self.limit;
            let radial = self.vel.dot(n);
            if radial > 0.0 {
                self.vel -= n * radial;
            }
        }
    }

    fn tether(&self, time: f32) -> Vec<LineVertex> {
        let a = self.anchor;
        let b = self.pos - Vec3::Y * EYE_HEIGHT_OFFSET;
        let side = (b - a).cross(Vec3::Y).normalize_or_zero();
        let mut pts = Vec::with_capacity(33);
        for i in 0..=32 {
            let t = i as f32 / 32.0;
            let sway = (std::f32::consts::PI * t).sin() * 0.06 * (time * 0.9).sin();
            pts.push(a.lerp(b, t) + side * sway);
        }
        let mut verts = Vec::with_capacity(64);
        for w in pts.windows(2) {
            verts.push(LineVertex {
                pos: w[0].to_array(),
                color: [0.95, 0.32, 0.45, 0.75],
            });
            verts.push(LineVertex {
                pos: w[1].to_array(),
                color: [0.95, 0.32, 0.45, 0.75],
            });
        }
        verts
    }
}

struct Launch {
    shot: Option<PathBuf>,
    pos: Vec3,
    yaw: f32,
    pitch: f32,
    proj: f32,
}

struct App {
    launch: Launch,
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    camera: Camera,
    keys: Keys,
    look: (f32, f32),
    grabbed: bool,
    last: Instant,
    start: Instant,
    frames: u32,
    fps_frames: u32,
    fps_t0: Instant,
    fullscreen: bool,
    proj: f32,
    proj_target: f32,
    proj_slow: f32,
}

impl App {
    fn new(launch: Launch) -> Self {
        let camera = Camera::new(launch.pos, launch.yaw, launch.pitch);
        let proj0 = launch.proj;
        Self {
            launch,
            window: None,
            renderer: None,
            camera,
            keys: Keys::default(),
            look: (0.0, 0.0),
            grabbed: false,
            last: Instant::now(),
            start: Instant::now(),
            frames: 0,
            fps_frames: 0,
            fps_t0: Instant::now(),
            fullscreen: false,
            proj: proj0,
            proj_target: proj0,
            proj_slow: proj0,
        }
    }

    fn toggle_projector(&mut self) {
        self.proj_target = if self.proj_target > 0.5 { 0.0 } else { 1.0 };
    }

    fn aiming_at_projector(&self) -> bool {
        let to = Vec3::from_array(geometry::PROJECTOR_POS) - self.camera.pos;
        let dir = self.camera.forward();
        let t = to.dot(dir);
        t > 0.0 && (to - dir * t).length() < geometry::PROJECTOR_RADIUS
    }

    fn grab(&mut self) {
        if let Some(window) = &self.window {
            let ok = window
                .set_cursor_grab(CursorGrabMode::Locked)
                .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined))
                .is_ok();
            if ok {
                window.set_cursor_visible(false);
                self.grabbed = true;
            }
        }
    }

    fn ungrab(&mut self) {
        if let Some(window) = &self.window {
            let _ = window.set_cursor_grab(CursorGrabMode::None);
            window.set_cursor_visible(true);
        }
        self.grabbed = false;
    }

    fn redraw(&mut self, event_loop: &ActiveEventLoop) {
        let hover = self.aiming_at_projector();
        let Some(renderer) = self.renderer.as_mut() else {
            return;
        };

        let now = Instant::now();
        let dt = (now - self.last).as_secs_f32().min(0.05);
        self.last = now;
        let time = (now - self.start).as_secs_f32();

        let (mx, my) = self.look;
        self.look = (0.0, 0.0);
        self.camera.update(dt, &self.keys, mx, my);
        self.proj += (self.proj_target - self.proj) * (1.0 - (-dt * 0.9).exp());
        // The planet dissolves on its own, slower curve than the light field.
        self.proj_slow += (self.proj_target - self.proj_slow) * (1.0 - (-dt * 0.35).exp());

        let (view_proj, view, proj) = self.camera.matrices(renderer.aspect());
        let mut tether = self.camera.tether(time);
        if hover {
            tether.extend(geometry::projector_outline());
        }

        if let Some(path) = self.launch.shot.clone() {
            self.frames += 1;
            if self.frames >= 6 {
                match renderer.screenshot(
                    &path,
                    view_proj,
                    view,
                    proj,
                    self.camera.pos,
                    time + 4.0,
                    self.proj,
                    self.proj_slow,
                    &tether,
                ) {
                    Ok(()) => println!("screenshot: {}", path.display()),
                    Err(e) => eprintln!("screenshot error: {e:#}"),
                }
                event_loop.exit();
            }
        } else if let Err(e) = renderer.render(
            view_proj,
            view,
            proj,
            self.camera.pos,
            time,
            self.proj,
            self.proj_slow,
            &tether,
        ) {
            eprintln!("render error: {e:#}");
        } else if std::env::var_os("PETROVE_LINE_FPS").is_some() {
            self.fps_frames += 1;
            let el = now.duration_since(self.fps_t0).as_secs_f32();
            if el > 2.0 {
                println!("FPS {:.1}", self.fps_frames as f32 / el);
                self.fps_frames = 0;
                self.fps_t0 = now;
            }
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        event_loop.set_control_flow(ControlFlow::Poll);

        let window = match event_loop.create_window(
            Window::default_attributes()
                .with_title(
                    "Petrova Line Simulator  |  Click: lock mouse  |  WASD + Space/Shift  |  P or click projector: lights  |  F11: fullscreen  |  Esc",
                )
                .with_inner_size(LogicalSize::new(1600.0, 900.0)),
        ) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                eprintln!("failed to create window: {e}");
                event_loop.exit();
                return;
            }
        };

        let renderer = match Renderer::new(window.clone()) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("failed to initialize renderer: {e:#}");
                event_loop.exit();
                return;
            }
        };

        self.renderer = Some(renderer);
        self.window = Some(window);
        self.last = Instant::now();
        self.start = Instant::now();
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(renderer) = self.renderer.as_mut() {
                    renderer.resize(size.width, size.height);
                }
            }
            WindowEvent::Focused(false) => self.ungrab(),
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => {
                if self.launch.shot.is_none() {
                    if self.grabbed {
                        if self.aiming_at_projector() {
                            self.toggle_projector();
                        }
                    } else {
                        self.grab();
                    }
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let pressed = event.state == ElementState::Pressed;
                if let PhysicalKey::Code(code) = event.physical_key {
                    match code {
                        KeyCode::Escape => {
                            if pressed {
                                if self.grabbed {
                                    self.ungrab();
                                } else {
                                    event_loop.exit();
                                }
                            }
                        }
                        KeyCode::KeyW => self.keys.w = pressed,
                        KeyCode::KeyA => self.keys.a = pressed,
                        KeyCode::KeyS => self.keys.s = pressed,
                        KeyCode::KeyD => self.keys.d = pressed,
                        KeyCode::Space => self.keys.up = pressed,
                        KeyCode::ShiftLeft | KeyCode::ShiftRight => self.keys.down = pressed,
                        KeyCode::KeyP => {
                            if pressed {
                                self.toggle_projector();
                            }
                        }
                        KeyCode::F11 => {
                            if pressed {
                                self.fullscreen = !self.fullscreen;
                                if let Some(window) = &self.window {
                                    window.set_fullscreen(if self.fullscreen {
                                        Some(Fullscreen::Borderless(None))
                                    } else {
                                        None
                                    });
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
            WindowEvent::RedrawRequested => self.redraw(event_loop),
            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        if let DeviceEvent::MouseMotion { delta } = event {
            if self.grabbed {
                self.look.0 += delta.0 as f32;
                self.look.1 += delta.1 as f32;
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

fn parse_launch() -> Launch {
    let args: Vec<String> = std::env::args().collect();
    let mut launch = Launch {
        shot: None,
        pos: Vec3::new(0.8, 6.2, -6.2),
        yaw: -10.0,
        pitch: -6.0,
        proj: 0.0,
    };
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--shot" => {
                i += 1;
                if i < args.len() {
                    launch.shot = Some(PathBuf::from(&args[i]));
                }
            }
            "--yaw" => {
                i += 1;
                if let Some(v) = args.get(i).and_then(|s| s.parse().ok()) {
                    launch.yaw = v;
                }
            }
            "--pitch" => {
                i += 1;
                if let Some(v) = args.get(i).and_then(|s| s.parse().ok()) {
                    launch.pitch = v;
                }
            }
            "--pos" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    let parts: Vec<f32> =
                        v.split(',').filter_map(|p| p.trim().parse().ok()).collect();
                    if parts.len() == 3 {
                        launch.pos = Vec3::new(parts[0], parts[1], parts[2]);
                    }
                }
            }
            "--proj" => {
                i += 1;
                if let Some(v) = args.get(i).and_then(|s| s.parse().ok()) {
                    launch.proj = v;
                }
            }
            _ => {}
        }
        i += 1;
    }
    launch
}

fn main() -> Result<()> {
    let launch = parse_launch();
    let event_loop = EventLoop::new()?;
    let mut app = App::new(launch);
    event_loop.run_app(&mut app)?;
    Ok(())
}
