# PCA Quick Start Guide

## 🚀 Quick Access

### Frontend
Navigate to: **http://localhost:5173/ml/pca**

### Backend API
- **Info**: `GET http://localhost:8000/api/ml/pca/info`
- **Train**: `POST http://localhost:8000/api/ml/pca/train`

## 📋 Quick Test

### 1. Test Backend (Python)
```bash
cd backend
python3 test_pca.py
```

Expected: All 3 tests pass ✓

### 2. Test API Endpoint (curl)
```bash
curl -X POST http://localhost:8000/api/ml/pca/train \
  -H "Content-Type: application/json" \
  -d '{
    "n_components": 2,
    "whiten": false,
    "random_state": 42
  }'
```

### 3. Get Algorithm Info
```bash
curl http://localhost:8000/api/ml/pca/info
```

## 🎯 What to Test

### Frontend Features
1. ✅ Adjust n_components slider (2-10)
2. ✅ Toggle whiten checkbox
3. ✅ Click "Train PCA" button
4. ✅ View scatter plot of first 2 PCs
5. ✅ Check explained variance bar chart
6. ✅ Read algorithm theory section

### Expected Results
- **Execution time**: ~15-50ms
- **Variance explained**: ~20-30% by PC1, ~15-20% by PC2
- **Total variance (2 PCs)**: ~35-45%
- **Samples**: 1,257 training samples
- **Original features**: 64 (8×8 pixel digits)

## 📊 Sample Output

### Scatter Plot
- 10 different colored clusters (one per digit: 0-9)
- Clear separation between some digit classes
- Some overlap expected (e.g., 3 and 8, 4 and 9)

### Variance Chart
- PC1: Highest bar (~20-30%)
- PC2: Second highest (~15-20%)
- Cumulative: Shows total variance captured

## 🔧 Files Modified/Created

### Backend
```
✅ backend/algorithms/ml/pca/__init__.py
✅ backend/algorithms/ml/pca/model.py
✅ backend/algorithms/ml/pca/data.py
✅ backend/algorithms/ml/pca/schema.py
✅ backend/api/routes/ml.py (updated)
✅ backend/test_pca.py
```

### Frontend
```
✅ frontend/src/pages/PCADemo.tsx
✅ frontend/src/App.tsx (updated)
```

## 🎓 Learning Points

### Algorithm Behavior
- **More components**: Capture more variance but harder to visualize
- **Whitening**: Makes components uncorrelated with unit variance
- **First PC**: Direction of maximum variance in data
- **Second PC**: Direction of second-most variance, orthogonal to first

### Dataset Insights
- Digits dataset is high-dimensional (64D)
- PCA reduces to 2D/3D for visualization
- Some digits naturally separate (e.g., 0, 1)
- Others are similar in pixel space (e.g., 4, 9)

## 🐛 Troubleshooting

### Backend Issues
```bash
# If imports fail, check scikit-learn installation
pip install scikit-learn==1.5.2

# If data loading fails, verify datasets utility
python3 -c "from utils.datasets import DatasetManager; print(DatasetManager.get_digits())"
```

### Frontend Issues
```bash
# If component doesn't render
npm install
npm run dev

# Check console for API errors
# Verify backend is running on port 8000
```

## 📝 Notes

- PCA automatically standardizes data (zero mean, unit variance)
- Original 64 features (8×8 pixels) reduced to 2-10 components
- Variance explained depends on intrinsic dimensionality of data
- More components generally needed for higher accuracy in downstream tasks

## 🔗 Related Routes

From ML page (`/ml`):
- Click on "Principal Component Analysis" card
- Or navigate directly to `/ml/pca`

## ⚡ Performance

- Training time: ~15-50ms (1,257 samples)
- Scales well for datasets up to ~10,000 samples
- For larger datasets, consider Incremental PCA

## 📚 Further Reading

- **Theory**: See "About PCA" section in the UI
- **Sklearn Docs**: https://scikit-learn.org/stable/modules/decomposition.html#pca
- **Related Algorithms**: t-SNE, LDA, Autoencoders
