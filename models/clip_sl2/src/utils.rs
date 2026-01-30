/// `SigLIP` requires a sigmoid activation on the dot product of embeddings
/// rather than a softmax used in standard CLIP.
#[must_use]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
