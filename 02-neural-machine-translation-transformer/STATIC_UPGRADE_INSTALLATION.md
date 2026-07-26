# Project 02 Static Upgrade Package

Merge this package into the existing `transformer-projects` repository. It
contains only:

- `.github/workflows/02-neural-machine-translation-transformer.yml`
- the complete `02-neural-machine-translation-transformer/` folder

Replace the existing Project 02 files when prompted. Projects 01 and 03–10 are
not included and will not be changed.

After merging, stage exactly these two paths:

```cmd
git add "02-neural-machine-translation-transformer" ".github/workflows/02-neural-machine-translation-transformer.yml"
git diff --cached --name-only
git commit -m "Upgrade Project 02 with free Hugging Face Static Space"
git push origin main
```
