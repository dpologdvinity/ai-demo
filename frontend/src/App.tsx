import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Layout from '@/components/layout/Layout';
import Home from '@/pages/Home';
import ML from '@/pages/ML';
import DeepLearning from '@/pages/DeepLearning';
import NLP from '@/pages/NLP';
import ComputerVision from '@/pages/ComputerVision';
import ReinforcementLearning from '@/pages/ReinforcementLearning';
import LinearRegressionDemo from '@/components/algorithm-demos/ml/LinearRegression';
import RidgeRegressionDemo from '@/components/algorithm-demos/ml/RidgeRegression';
import LassoRegressionDemo from '@/components/algorithm-demos/ml/LassoRegression';
import ElasticNetDemo from '@/components/algorithm-demos/ml/ElasticNet';
import XGBoostDemo from '@/components/algorithm-demos/ml/XGBoost';
import DecisionTreeDemo from '@/components/algorithm-demos/ml/DecisionTree';
import KMeansDemo from '@/components/algorithm-demos/k-means';
import DBSCANDemo from '@/components/algorithm-demos/dbscan';
import LogisticRegressionDemo from '@/components/algorithm-demos/logistic-regression';
import RandomForestDemo from '@/components/algorithm-demos/ml/RandomForest';
import KNNDemo from '@/components/algorithm-demos/ml/KNN';
import NaiveBayesDemo from '@/components/algorithm-demos/ml/NaiveBayes';
import HierarchicalClusteringDemo from '@/components/algorithm-demos/ml/HierarchicalClustering';
import SVMDemo from '@/components/algorithm-demos/ml/SVM';
import PCADemo from '@/pages/PCADemo';
import GMMDemo from '@/components/algorithm-demos/gmm';
import TSNEDemo from '@/components/algorithm-demos/ml/TSNE';
import SentimentAnalysisDemo from '@/components/algorithm-demos/nlp/SentimentAnalysis';

function App() {
  return (
    <Router>
      <Layout>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/ml" element={<ML />} />
          <Route path="/ml/linear-regression" element={<LinearRegressionDemo />} />
          <Route path="/ml/ridge-regression" element={<RidgeRegressionDemo />} />
          <Route path="/ml/lasso-regression" element={<LassoRegressionDemo />} />
          <Route path="/ml/elastic-net" element={<ElasticNetDemo />} />
          <Route path="/ml/logistic-regression" element={<LogisticRegressionDemo />} />
          <Route path="/ml/decision-tree" element={<DecisionTreeDemo />} />
          <Route path="/ml/xgboost" element={<XGBoostDemo />} />
          <Route path="/ml/k-means" element={<KMeansDemo />} />
          <Route path="/ml/dbscan" element={<DBSCANDemo />} />
          <Route path="/ml/random-forest" element={<RandomForestDemo />} />
          <Route path="/ml/knn" element={<KNNDemo />} />
          <Route path="/ml/naive-bayes" element={<NaiveBayesDemo />} />
          <Route path="/ml/svm" element={<SVMDemo />} />
          <Route path="/ml/hierarchical-clustering" element={<HierarchicalClusteringDemo />} />
          <Route path="/ml/gmm" element={<GMMDemo />} />
          <Route path="/ml/pca" element={<PCADemo />} />
          <Route path="/ml/tsne" element={<TSNEDemo />} />
          <Route path="/deep-learning" element={<DeepLearning />} />
          <Route path="/nlp" element={<NLP />} />
          <Route path="/nlp/sentiment-analysis" element={<SentimentAnalysisDemo />} />
          <Route path="/computer-vision" element={<ComputerVision />} />
          <Route path="/reinforcement-learning" element={<ReinforcementLearning />} />
        </Routes>
      </Layout>
    </Router>
  );
}

export default App;
