# GitHub Push Guide - Dataset and File Size Management

## 📊 **Folder Sizes Summary**

Based on analysis of your project:

| Folder | Size | Push to GitHub? |
|--------|------|-----------------|
| **Dataset/** | 62 MB | ⚠️ Optional (large) |
| **Wheel_seg-6/** | 12 MB | ⚠️ Optional (medium) |
| **reconstructed_videos/** | 123 MB | ❌ NO (too large) |
| **demo_outputs/** | 120 MB | ❌ NO (too large) |
| **runs/** | 45 MB | ⚠️ Partial (models only) |

---

## ⚠️ **GitHub Limitations**

- **Maximum file size:** 100 MB per file
- **Recommended repo size:** < 1 GB
- **Warning threshold:** > 50 MB per file

**Problem:** Your videos exceed GitHub limits!

---

## ✅ **Recommended GitHub Strategy**

### **Option 1: Essential Files Only** (Recommended for Thesis)

**PUSH to GitHub:**
```
✅ Code Files (~1 MB)
├── *.py (all Python scripts)
├── *.sh (shell scripts)
├── *.md (documentation)
├── *.json (config files)
├── *.txt (text files)
├── requirements.txt
└── .gitignore

✅ Configuration
├── config.py
├── gopro_calibration_fisheye.npz (1.3 KB)
├── coordinate_mapping*.json
└── Ground Truth Data.docx

✅ Small Model Files
└── runs/segment/wheel_seg/weights/
    └── best.pt (19 MB) ✓ Under limit
```

**DO NOT PUSH:**
```
❌ Dataset/ (62 MB) - Available from Roboflow
❌ Wheel_seg-6/ (12 MB) - Available from Roboflow
❌ reconstructed_videos/ (123 MB) - Can regenerate
❌ demo_outputs/ (120 MB) - Can regenerate
❌ runs/segment/wheel_seg/*.png - Can regenerate
❌ test_output_*.jpg - Can regenerate
```

**Total Size:** ~25-30 MB ✓ Perfect!

---

### **Option 2: Include Datasets** (If needed for reproducibility)

If you want to include datasets:

**Use Git LFS (Large File Storage):**
```bash
# Install Git LFS
brew install git-lfs  # macOS
git lfs install

# Track large files
git lfs track "Dataset/**/*"
git lfs track "Wheel_seg-6/**/*"
git lfs track "*.pt"

# Add .gitattributes
git add .gitattributes
```

**Note:** Git LFS has bandwidth limits (1 GB/month free)

---

### **Option 3: External Storage** (Best for Large Files)

Store large files externally and provide download links:

**Services:**
1. **Google Drive** - Unlimited (with university account)
2. **Roboflow** - Your datasets are already there!
3. **Kaggle Datasets** - Free, public datasets
4. **Zenodo** - Academic data repository (free)
5. **Hugging Face** - For models and datasets

**In your README.md:**
```markdown
## Datasets

Download from:
- **Vehicle Dataset:** [Roboflow - car-bb/5](https://universe.roboflow.com/...)
- **Wheel Segmentation:** [Roboflow - wheel_seg/6](https://universe.roboflow.com/...)
- **Trained Model:** [Google Drive](https://drive.google.com/...)
- **Demo Videos:** [Google Drive](https://drive.google.com/...)
```

---

## 📝 **Create .gitignore File**

```bash
cat > .gitignore << 'EOF'
# Large datasets (available from Roboflow)
Dataset/
Wheel_seg-6/

# Generated videos (can regenerate)
reconstructed_videos/
demo_outputs/

# Training outputs (except model weights)
runs/detect/
runs/segment/wheel_seg/*.png
runs/segment/wheel_seg/*.jpg
runs/segment/wheel_seg/results.csv

# Test outputs (can regenerate)
test_output_*.jpg

# Logs
*.log
nohup.out
auto_demo.log
overnight_execution.log

# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/
.pytest_cache/
.coverage

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Temporary files
*.tmp
*.bak
pipeline_summary.json
EOF
```

---

## 🚀 **Recommended Git Workflow**

### Step 1: Create .gitignore
```bash
cat > .gitignore << 'EOF'
# Datasets (download from Roboflow)
Dataset/
Wheel_seg-6/

# Generated content
reconstructed_videos/
demo_outputs/
test_output_*.jpg
*.log

# Training artifacts (keep only model)
runs/detect/
runs/segment/wheel_seg/*.png
runs/segment/wheel_seg/results.csv

# Python
__pycache__/
*.pyc

# OS
.DS_Store
EOF
```

### Step 2: Check What Will Be Pushed
```bash
git add .
git status
# Review the list - should be ~20-30 MB total
```

### Step 3: Commit and Push
```bash
git commit -m "Add vehicle speed estimation pipeline

- YOLOv8 wheel segmentation (95.1% mAP@50)
- Complete demo pipeline
- Camera calibration data
- Documentation and reports
"

git push origin main
```

---

## 📋 **What Should Be in Your GitHub Repo**

### Essential (MUST include):
```
✅ All Python scripts (.py files)
✅ Documentation (.md files)
✅ Configuration files (.json, .npz)
✅ Requirements.txt
✅ Trained model (best.pt - 19 MB)
✅ README.md with dataset download links
```

### Optional (if space allows):
```
⚠️ Sample images from datasets (few examples)
⚠️ Training results CSV
⚠️ Sample output images (1-2 examples)
```

### Exclude (too large):
```
❌ Full datasets (62 MB + 12 MB)
❌ Videos (123 MB + 120 MB)
❌ All training checkpoints
```

---

## 📄 **Create README.md for GitHub**

```markdown
# Vehicle Speed Estimation using Wheel Segmentation

YOLOv8-based wheel segmentation and tracking for vehicle speed estimation.

## Performance

- **Wheel Detection:** 95.1% mAP@50
- **Model:** YOLOv8n-seg
- **Dataset:** 1,420 annotated images (3 wheel classes)

## Installation

\`\`\`bash
pip install -r requirements.txt
\`\`\`

## Datasets

Download datasets from Roboflow:
- [Vehicle Dataset (car-bb/5)](https://universe.roboflow.com/rptu-av2mn/car-bb/dataset/5)
- [Wheel Segmentation (wheel_seg/6)](https://universe.roboflow.com/rptu-av2mn/wheel_seg/dataset/6)

Place in:
- \`Dataset/\` - Vehicle detection dataset
- \`Wheel_seg-6/\` - Wheel segmentation dataset

## Trained Model

Download trained model:
- [Google Drive - best.pt (19 MB)](https://drive.google.com/...)

Or train yourself:
\`\`\`bash
python train_wheel_segmentation.py
\`\`\`

## Usage

### Run Demo
\`\`\`bash
python demo_wheel_pipeline.py
\`\`\`

### Validate (with test videos)
\`\`\`bash
python validate_speed_estimation.py --auto
\`\`\`

## Project Structure

\`\`\`
├── demo_wheel_pipeline.py      # Main demo
├── train_wheel_segmentation.py # Training
├── validate_speed_estimation.py # Validation
├── gopro_calibration_fisheye.npz # Camera calibration
└── runs/segment/wheel_seg/weights/best.pt # Trained model
\`\`\`

## Results

See \`COMPLETE_PIPELINE_REPORT.md\` for detailed results.

## Citation

If you use this code, please cite:
\`\`\`
[Your thesis citation]
\`\`\`
```

---

## 🎯 **Quick Commands**

### Check Current Repo Size
```bash
du -sh .git/
```

### Check What Will Be Committed
```bash
git add .
git status
git diff --stat --cached
```

### Dry Run (see what would be pushed)
```bash
git add .
git status | grep "new file"
```

### Actually Push
```bash
git add .
git commit -m "Complete vehicle speed estimation pipeline"
git push origin main
```

---

## 💡 **Pro Tips**

1. **Always check `git status` before pushing**
   - Verify no large files are staged

2. **Use meaningful commit messages**
   - Describe what the code does
   - Mention key results (e.g., "95.1% mAP@50")

3. **Include a good README.md**
   - Installation instructions
   - Dataset download links
   - Model download link
   - Usage examples

4. **Consider GitHub Releases**
   - Upload model as a release asset
   - Attach demo videos to releases
   - Version your models

5. **Update README with badges**
   - ![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
   - ![Model](https://img.shields.io/badge/model-YOLOv8n--seg-green.svg)
   - ![mAP@50](https://img.shields.io/badge/mAP@50-95.1%25-brightgreen.svg)

---

## ⚠️ **Common Mistakes to Avoid**

❌ Pushing datasets without .gitignore
❌ Committing large video files
❌ Uploading training checkpoints
❌ Including log files
❌ Forgetting .DS_Store (macOS)

✅ Use .gitignore properly
✅ Keep repo lean (< 100 MB)
✅ Provide download links for large files
✅ Include comprehensive README

---

## 📊 **Estimated Sizes After .gitignore**

| What's Pushed | Size |
|---------------|------|
| Python scripts | ~500 KB |
| Documentation | ~200 KB |
| Config files | ~50 KB |
| Trained model | ~19 MB |
| **TOTAL** | **~20 MB** ✓ |

Perfect for GitHub! 🎉

---

## 🚀 **Ready to Push?**

### Final Checklist:

- [ ] Created .gitignore
- [ ] Verified file sizes (`du -sh *`)
- [ ] Checked git status
- [ ] Wrote good README.md
- [ ] Added dataset download links
- [ ] Tested `git add .` (check what's staged)
- [ ] Ready to commit and push!

---

**Good to go! Your repository will be clean and professional!** ✨
