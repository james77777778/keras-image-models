# Convert Model Weights

## Convert the weights from `timm` and `keras`

- Use TensorFlow backend

```bash
# At project root ./kimm/
./shell/export_models.sh
```

## Upload to Releases

Setup `gh`

[https://github.com/cli/cli/blob/trunk/docs/install_linux.md](https://github.com/cli/cli/blob/trunk/docs/install_linux.md)

Upload the converted file

```bash
gh release upload <tag> <files>... [flags]

# For example:
gh release upload 0.1.0 exported/visiontransformertiny16_vit_tiny_patch16_384.keras
```
