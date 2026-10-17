//! Serialization and deserialization to transfer between Ortex and BinaryBackend
//! [Nx](https://hexdocs.pm/nx) backend.

use crate::constants::*;
use crate::tensor::OrtexTensor;
use ndarray::{ArrayViewMut, Ix, IxDyn};

use ndarray::ShapeError;

use rustler::resource::ResourceArc;
use rustler::types::Binary;
use rustler::{Atom, Env, NifResult};

use ort::{ExecutionProviderDispatch, GraphOptimizationLevel};

/// A faster (unsafe) way of creating an Array from an Erlang binary
fn initialize_from_raw_ptr<T>(ptr: *const T, shape: &[Ix]) -> ArrayViewMut<T, IxDyn> {
    let array = unsafe { ArrayViewMut::from_shape_ptr(shape, ptr as *mut T) };
    array
}

/// Given a Binary term, shape, and dtype from the BEAM, constructs an OrtexTensor and
/// returns the reference to be used as an Nx.Backend representation.
///
/// # Example
///
/// ```elixir
/// bin = <<1, 0, 0, 0, 1, 0, 0, 0>>
/// ```
///
/// Create a shape `[2]` u32 OrtexTensor from a binary of 8 bytes
/// ```elixir
/// {:ok, reference} = from_binary(bin, {2}, {:u, 32})
/// ```
pub fn from_binary(
    bin: Binary,
    shape: Vec<usize>,
    dtype_str: String,
    dtype_bits: usize,
) -> Result<ResourceArc<OrtexTensor>, ShapeError> {
    match (dtype_str.as_ref(), dtype_bits) {
        ("bf", 16) => Ok(ResourceArc::new(OrtexTensor::bf16(
            initialize_from_raw_ptr(bin.as_ptr() as *const half::bf16, &shape).to_owned(),
        ))),
        ("f", 16) => Ok(ResourceArc::new(OrtexTensor::f16(
            initialize_from_raw_ptr(bin.as_ptr() as *const half::f16, &shape).to_owned(),
        ))),
        ("f", 32) => Ok(ResourceArc::new(OrtexTensor::f32(
            initialize_from_raw_ptr(bin.as_ptr() as *const f32, &shape).to_owned(),
        ))),
        ("f", 64) => Ok(ResourceArc::new(OrtexTensor::f64(
            initialize_from_raw_ptr(bin.as_ptr() as *const f64, &shape).to_owned(),
        ))),
        ("s", 8) => Ok(ResourceArc::new(OrtexTensor::s8(
            initialize_from_raw_ptr(bin.as_ptr() as *const i8, &shape).to_owned(),
        ))),
        ("s", 16) => Ok(ResourceArc::new(OrtexTensor::s16(
            initialize_from_raw_ptr(bin.as_ptr() as *const i16, &shape).to_owned(),
        ))),
        ("s", 32) => Ok(ResourceArc::new(OrtexTensor::s32(
            initialize_from_raw_ptr(bin.as_ptr() as *const i32, &shape).to_owned(),
        ))),
        ("s", 64) => Ok(ResourceArc::new(OrtexTensor::s64(
            initialize_from_raw_ptr(bin.as_ptr() as *const i64, &shape).to_owned(),
        ))),
        ("u", 8) => Ok(ResourceArc::new(OrtexTensor::u8(
            initialize_from_raw_ptr(bin.as_ptr() as *const u8, &shape).to_owned(),
        ))),
        ("u", 16) => Ok(ResourceArc::new(OrtexTensor::u16(
            initialize_from_raw_ptr(bin.as_ptr() as *const u16, &shape).to_owned(),
        ))),
        ("u", 32) => Ok(ResourceArc::new(OrtexTensor::u32(
            initialize_from_raw_ptr(bin.as_ptr() as *const u32, &shape).to_owned(),
        ))),
        ("u", 64) => Ok(ResourceArc::new(OrtexTensor::u64(
            initialize_from_raw_ptr(bin.as_ptr() as *const u64, &shape).to_owned(),
        ))),
        (&_, _) => unimplemented!(),
    }
}

/// Given a reference to an OrtexTensor return the binary representation to be used
/// by the BinaryBackend of Nx.
pub fn to_binary<'a>(
    env: Env<'a>,
    reference: ResourceArc<OrtexTensor>,
    _bits: usize,
    _limit: usize,
) -> NifResult<Binary<'a>> {
    Ok(reference.make_binary(env, |x| x.to_bytes()))
}

/// Whether `eps` asks for CoreML, which `model::init` registers itself, as an
/// ML Program on the GPU (see `model::append_coreml`), when the `coreml`
/// feature is on.
pub fn wants_coreml(env: rustler::env::Env, eps: &[Atom]) -> bool {
    cfg!(feature = "coreml")
        && eps
            .iter()
            .any(|e| e.to_term(env).atom_to_string().unwrap() == COREML)
}

/// Takes a vec of Atoms and transforms them into a vec of ExecutionProvider Enums.
/// With the `coreml` feature on, CoreML is left out: `model::init` registers it.
pub fn map_eps(env: rustler::env::Env, eps: Vec<Atom>) -> Vec<ExecutionProviderDispatch> {
    eps.iter()
        .filter(|e| !(cfg!(feature = "coreml") && e.to_term(env).atom_to_string().unwrap() == COREML))
        .map(|e| match &e.to_term(env).atom_to_string().unwrap()[..] {
            CPU => ort::CPUExecutionProvider::default().build(),
            CUDA => ort::CUDAExecutionProvider::default().build(),
            TENSORRT => ort::TensorRTExecutionProvider::default().build(),
            ACL => ort::ACLExecutionProvider::default().build(),
            ONEDNN => ort::OneDNNExecutionProvider::default().build(),
            COREML => ort::CoreMLExecutionProvider::default().build(),
            DIRECTML => ort::DirectMLExecutionProvider::default().build(),
            ROCM => ort::ROCmExecutionProvider::default().build(),
            _ => ort::CPUExecutionProvider::default().build(),
        })
        .collect()
}

/// Take an optimization level and returns the
pub fn map_opt_level(opt: i32) -> GraphOptimizationLevel {
    match opt {
        1 => GraphOptimizationLevel::Level1,
        2 => GraphOptimizationLevel::Level2,
        3 => GraphOptimizationLevel::Level3,
        _ => GraphOptimizationLevel::Disable,
    }
}

pub fn is_bool_input(inp: &ort::ValueType) -> bool {
    match inp {
        ort::ValueType::Tensor { ty, .. } => ty == &ort::TensorElementType::Bool,
        ort::ValueType::Map { value, .. } => value == &ort::TensorElementType::Bool,
        ort::ValueType::Sequence(boxed_input) => is_bool_input(boxed_input),
        ort::ValueType::Optional(boxed_input) => is_bool_input(boxed_input),
    }
}

/// Calls ONNX Runtime's `GetAvailableProviders` on the loaded library and
/// returns its list, in the runtime's own order and spelling
/// (`CPUExecutionProvider`, `CUDAExecutionProvider`, ...).
///
/// `ort`'s `ExecutionProvider::is_available` makes the same call, but answers
/// for one provider at a time and only for providers `ort` knows by name.
pub fn available_providers() -> Result<Vec<String>, String> {
    let api = unsafe { ort::api().as_ref() };
    let get = api
        .GetAvailableProviders
        .ok_or("GetAvailableProviders is null")?;
    let release = api
        .ReleaseAvailableProviders
        .ok_or("ReleaseAvailableProviders is null")?;

    let mut providers: *mut *mut std::os::raw::c_char = std::ptr::null_mut();
    let mut count: std::os::raw::c_int = 0;
    let status = unsafe { get(&mut providers, &mut count) };
    if !status.is_null() {
        return Err(status_message(api, status));
    }
    if providers.is_null() {
        return Ok(Vec::new());
    }

    let names = (0..count as isize)
        .map(|i| {
            let name = unsafe { std::ffi::CStr::from_ptr(*providers.offset(i)) };
            name.to_string_lossy().into_owned()
        })
        .collect();

    let status = unsafe { release(providers, count) };
    if !status.is_null() {
        return Err(status_message(api, status));
    }
    Ok(names)
}

fn status_message(api: &ort::sys::OrtApi, status: *mut ort::sys::OrtStatus) -> String {
    let message = match api.GetErrorMessage {
        Some(get) => unsafe { std::ffi::CStr::from_ptr(get(status)) }
            .to_string_lossy()
            .into_owned(),
        None => "ONNX Runtime returned an error status".to_string(),
    };
    if let Some(release) = api.ReleaseStatus {
        unsafe { release(status) };
    }
    message
}
