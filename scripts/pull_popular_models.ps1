$models = @(
#    "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", # ✅
#    "redlessone/DermLIP_ViT-B-16", # ✅
#    "timm/MobileCLIP2-S2-OpenCLIP",#✅
#    "timm/MobileCLIP2-S3-OpenCLIP",#✅
#    "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",#✅
#    "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",#✅
#    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",#✅
#    "laion/CLIP-ViT-B-16-laion2B-s34B-b88K",#✅
#    "laion/CLIP-convnext_base_w-laion2B-s13B-b82K",#✅
#    "laion/CLIP-convnext_base_w_320-laion_aesthetic-s13B-b82K-augreg",#✅
#    "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",#✅
#    "laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft",#✅
#    "laion/CLIP-ViT-g-14-laion2B-s34B-b88K",#✅
#    "timm/vit_large_patch14_clip_336.openai",#✅
#    "laion/CLIP-ViT-B-32-CommonPool.S-s13M-b4K",#✅
#    "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K",#✅
#    "timm/ViT-SO400M-14-SigLIP-384",#✅
#    "Marqo/marqo-fashionSigLIP",#✅
#    "imageomics/bioclip",#✅
#    "timm/ViT-B-16-SigLIP",#✅
#    "timm/ViT-L-16-SigLIP-384",#✅
#    "Marqo/marqo-fashionCLIP",#✅
#    "imageomics/bioclip-2",#✅
#    "wisdomik/QuiltNet-B-32",#✅
#    "timm/ViT-SO400M-14-SigLIP",#✅
#    "UCSC-VLAA/ViT-L-16-HTxt-Recap-CLIP",#✅
#    "visheratin/nllb-siglip-mrl-large",#⛔
#    "mkaichristensen/echo-clip",#✅
#    "visheratin/nllb-siglip-mrl-base",#⛔
#    "redlessone/DermLIP_PanDerm-base-w-PubMed-256",#⛔
#    "DatologyAI/cls-opt-vit-b-32",#✅
#    "laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K",#✅
#    "timm/vit_base_patch16_plus_clip_240.laion400m_e31",#✅
#    "timm/ViT-B-16-SigLIP2-256",#✅
#    "timm/vit_base_patch32_clip_224.laion400m_e32",#✅
#    "timm/ViT-B-16-SigLIP-256",#✅
#    "timm/resnet50_clip.openai",#✅
#    "timm/ViT-SO400M-14-SigLIP2",#✅
#    "timm/ViT-B-16-SigLIP2-512",#✅
#    "timm/PE-Core-bigG-14-448"#✅
)

$failed = @()

foreach ($model in $models)
{
    Write-Host "Processing: $model" -ForegroundColor Cyan
    uv run .\pull_onnx.py --id $model

    if ($LASTEXITCODE -ne 0)
    {
        $failed += $model
        Write-Host "FAILED: $model" -ForegroundColor Red
    }
}

if ($failed.Count -gt 0)
{
    Write-Host "`nSummary of failures:" -ForegroundColor Red
    $failed | ForEach-Object { Write-Host " - $_" }
}
else
{
    Write-Host "`nAll models processed successfully!" -ForegroundColor Green
}