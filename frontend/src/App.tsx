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

// Computer Vision demos
import EdgeDetectionDemo from '@/components/algorithm-demos/computer-vision/EdgeDetection';
import FaceDetectionDemo from '@/components/algorithm-demos/computer-vision/FaceDetection';
import ImageClassificationDemo from '@/components/algorithm-demos/computer-vision/ImageClassification';
import InstanceSegmentationDemo from '@/components/algorithm-demos/computer-vision/InstanceSegmentation';
import OpticalFlowDemo from '@/components/algorithm-demos/computer-vision/OpticalFlow';
import SemanticSegmentationDemo from '@/components/algorithm-demos/computer-vision/SemanticSegmentation';
import SiftDemo from '@/components/algorithm-demos/computer-vision/Sift';
import StyleTransferDemo from '@/components/algorithm-demos/computer-vision/StyleTransfer';
import YoloDetectionDemo from '@/components/algorithm-demos/computer-vision/YoloDetection';

// Deep Learning demos
import ActivationFunctionsDemo from '@/components/algorithm-demos/deep-learning/ActivationFunctions';
import AdamOptimizerDemo from '@/components/algorithm-demos/deep-learning/AdamOptimizer';
import AutoencoderDemo from '@/components/algorithm-demos/deep-learning/Autoencoder';
import AutoencoderVariantsDemo from '@/components/algorithm-demos/deep-learning/AutoencoderVariants';
import BatchNormalizationDemo from '@/components/algorithm-demos/deep-learning/BatchNormalization';
import CnnDemo from '@/components/algorithm-demos/deep-learning/Cnn';
import ConvolutionalLayersDemo from '@/components/algorithm-demos/deep-learning/ConvolutionalLayers';
import DataAugmentationDemo from '@/components/algorithm-demos/deep-learning/DataAugmentation';
import DropoutDemo from '@/components/algorithm-demos/deep-learning/Dropout';
import FeedforwardNnDemo from '@/components/algorithm-demos/deep-learning/FeedforwardNn';
import GanDemo from '@/components/algorithm-demos/deep-learning/Gan';
import GradientDescentDemo from '@/components/algorithm-demos/deep-learning/GradientDescent';
import GruDemo from '@/components/algorithm-demos/deep-learning/Gru';
import LearningRateSchedulingDemo from '@/components/algorithm-demos/deep-learning/LearningRateScheduling';
import LstmDemo from '@/components/algorithm-demos/deep-learning/Lstm';
import PoolingLayersDemo from '@/components/algorithm-demos/deep-learning/PoolingLayers';
import ResnetDemo from '@/components/algorithm-demos/deep-learning/Resnet';
import RnnDemo from '@/components/algorithm-demos/deep-learning/Rnn';
import TransferLearningDemo from '@/components/algorithm-demos/deep-learning/TransferLearning';
import TransformerDemo from '@/components/algorithm-demos/deep-learning/Transformer';
import VaeDemo from '@/components/algorithm-demos/deep-learning/Vae';
import VggDemo from '@/components/algorithm-demos/deep-learning/Vgg';

// Reinforcement Learning demos
import A3CDemo from '@/components/algorithm-demos/reinforcement-learning/A3c';
import ActorCriticDemo from '@/components/algorithm-demos/reinforcement-learning/ActorCritic';
import DQNDemo from '@/components/algorithm-demos/reinforcement-learning/DQN';
import DDPGDemo from '@/components/algorithm-demos/reinforcement-learning/Ddpg';
import PPODemo from '@/components/algorithm-demos/reinforcement-learning/Ppo';
import QLearningDemo from '@/components/algorithm-demos/reinforcement-learning/QLearning';
import SARSADemo from '@/components/algorithm-demos/reinforcement-learning/SARSA';

// NLP demos
import BagOfWordsDemo from '@/components/algorithm-demos/nlp/BagOfWords';
import BertFinetuningDemo from '@/components/algorithm-demos/nlp/BertFinetuning';
import GloVeDemo from '@/components/algorithm-demos/nlp/GloVe';
import NERDemo from '@/components/algorithm-demos/nlp/NER';
import POSTaggingDemo from '@/components/algorithm-demos/nlp/PosTagging';
import Seq2SeqDemo from '@/components/algorithm-demos/nlp/Seq2Seq';
import TFIDFDemo from '@/components/algorithm-demos/nlp/TFIDF';
import TextClassificationDemo from '@/components/algorithm-demos/nlp/TextClassification';
import TokenizationDemo from '@/components/algorithm-demos/nlp/Tokenization';
import TopicModelingDemo from '@/components/algorithm-demos/nlp/TopicModeling';
import Word2VecDemo from '@/components/algorithm-demos/nlp/Word2Vec';

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
          <Route path="/deep-learning/activation-functions" element={<ActivationFunctionsDemo />} />
          <Route path="/deep-learning/adam-optimizer" element={<AdamOptimizerDemo />} />
          <Route path="/deep-learning/autoencoder" element={<AutoencoderDemo />} />
          <Route path="/deep-learning/autoencoder-variants" element={<AutoencoderVariantsDemo />} />
          <Route path="/deep-learning/batch-normalization" element={<BatchNormalizationDemo />} />
          <Route path="/deep-learning/cnn" element={<CnnDemo />} />
          <Route path="/deep-learning/convolutional-layers" element={<ConvolutionalLayersDemo />} />
          <Route path="/deep-learning/data-augmentation" element={<DataAugmentationDemo />} />
          <Route path="/deep-learning/dropout" element={<DropoutDemo />} />
          <Route path="/deep-learning/feedforward-nn" element={<FeedforwardNnDemo />} />
          <Route path="/deep-learning/gan" element={<GanDemo />} />
          <Route path="/deep-learning/gradient-descent" element={<GradientDescentDemo />} />
          <Route path="/deep-learning/gru" element={<GruDemo />} />
          <Route path="/deep-learning/learning-rate-scheduling" element={<LearningRateSchedulingDemo />} />
          <Route path="/deep-learning/lstm" element={<LstmDemo />} />
          <Route path="/deep-learning/pooling-layers" element={<PoolingLayersDemo />} />
          <Route path="/deep-learning/resnet" element={<ResnetDemo />} />
          <Route path="/deep-learning/rnn" element={<RnnDemo />} />
          <Route path="/deep-learning/transfer-learning" element={<TransferLearningDemo />} />
          <Route path="/deep-learning/transformer" element={<TransformerDemo />} />
          <Route path="/deep-learning/vae" element={<VaeDemo />} />
          <Route path="/deep-learning/vgg" element={<VggDemo />} />
          <Route path="/nlp" element={<NLP />} />
          <Route path="/nlp/sentiment-analysis" element={<SentimentAnalysisDemo />} />
          <Route path="/nlp/bag-of-words" element={<BagOfWordsDemo />} />
          <Route path="/nlp/bert-finetuning" element={<BertFinetuningDemo />} />
          <Route path="/nlp/glove" element={<GloVeDemo />} />
          <Route path="/nlp/ner" element={<NERDemo />} />
          <Route path="/nlp/pos-tagging" element={<POSTaggingDemo />} />
          <Route path="/nlp/seq2seq" element={<Seq2SeqDemo />} />
          <Route path="/nlp/tfidf" element={<TFIDFDemo />} />
          <Route path="/nlp/text-classification" element={<TextClassificationDemo />} />
          <Route path="/nlp/tokenization" element={<TokenizationDemo />} />
          <Route path="/nlp/topic-modeling" element={<TopicModelingDemo />} />
          <Route path="/nlp/word2vec" element={<Word2VecDemo />} />
          <Route path="/computer-vision" element={<ComputerVision />} />
          <Route path="/computer-vision/edge-detection" element={<EdgeDetectionDemo />} />
          <Route path="/computer-vision/face-detection" element={<FaceDetectionDemo />} />
          <Route path="/computer-vision/image-classification" element={<ImageClassificationDemo />} />
          <Route path="/computer-vision/instance-segmentation" element={<InstanceSegmentationDemo />} />
          <Route path="/computer-vision/optical-flow" element={<OpticalFlowDemo />} />
          <Route path="/computer-vision/semantic-segmentation" element={<SemanticSegmentationDemo />} />
          <Route path="/computer-vision/sift" element={<SiftDemo />} />
          <Route path="/computer-vision/style-transfer" element={<StyleTransferDemo />} />
          <Route path="/computer-vision/yolo-detection" element={<YoloDetectionDemo />} />
          <Route path="/reinforcement-learning" element={<ReinforcementLearning />} />
          <Route path="/reinforcement-learning/a3c" element={<A3CDemo />} />
          <Route path="/reinforcement-learning/actor-critic" element={<ActorCriticDemo />} />
          <Route path="/reinforcement-learning/dqn" element={<DQNDemo />} />
          <Route path="/reinforcement-learning/ddpg" element={<DDPGDemo />} />
          <Route path="/reinforcement-learning/ppo" element={<PPODemo />} />
          <Route path="/reinforcement-learning/q-learning" element={<QLearningDemo />} />
          <Route path="/reinforcement-learning/sarsa" element={<SARSADemo />} />
        </Routes>
      </Layout>
    </Router>
  );
}

export default App;
