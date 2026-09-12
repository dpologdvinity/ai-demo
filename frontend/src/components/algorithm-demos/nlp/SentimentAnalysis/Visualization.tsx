import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Legend,
  Tooltip,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
} from 'recharts';

interface SentimentPrediction {
  text: string;
  sentiment: string;
  confidence: number;
  compound: number;
  scores: {
    positive: number;
    negative: number;
    neutral: number;
  };
}

interface AnalysisResult {
  success: boolean;
  predictions: SentimentPrediction[];
  distribution: {
    positive: number;
    negative: number;
    neutral: number;
    positive_pct: number;
    negative_pct: number;
    neutral_pct: number;
  };
  top_positive: SentimentPrediction[];
  top_negative: SentimentPrediction[];
  visualization_data: {
    pie_chart: {
      labels: string[];
      values: number[];
      percentages: number[];
      colors: string[];
    };
  };
}

interface VisualizationProps {
  result: AnalysisResult;
}

export function Visualization({ result }: VisualizationProps) {
  // Prepare pie chart data
  const pieData = result.visualization_data.pie_chart.labels.map((label, idx) => ({
    name: label,
    value: result.visualization_data.pie_chart.values[idx],
    percentage: result.visualization_data.pie_chart.percentages[idx],
  }));

  const COLORS = result.visualization_data.pie_chart.colors;

  // Prepare bar chart data for top predictions
  const topPredictionsData = [
    ...result.top_positive.slice(0, 3),
    ...result.top_negative.slice(0, 3),
  ].map((pred, idx) => ({
    name: `Text ${idx + 1}`,
    compound: pred.compound,
    sentiment: pred.sentiment,
  }));

  const getSentimentColor = (sentiment: string) => {
    switch (sentiment) {
      case 'positive':
        return 'bg-green-500';
      case 'negative':
        return 'bg-red-500';
      case 'neutral':
        return 'bg-gray-500';
      default:
        return 'bg-gray-400';
    }
  };

  const getSentimentBadgeVariant = (sentiment: string): "default" | "secondary" | "destructive" | "outline" => {
    switch (sentiment) {
      case 'positive':
        return 'default';
      case 'negative':
        return 'destructive';
      case 'neutral':
        return 'secondary';
      default:
        return 'outline';
    }
  };

  return (
    <div className="space-y-6">
      {/* Sentiment Distribution Pie Chart */}
      <Card>
        <CardHeader>
          <CardTitle>Sentiment Distribution</CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={pieData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ percentage }) => `${percentage.toFixed(1)}%`}
                outerRadius={100}
                fill="#8884d8"
                dataKey="value"
              >
                {pieData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip
                formatter={(value: number, name: string, props: any) => [
                  `${value} (${props.payload.percentage.toFixed(1)}%)`,
                  name,
                ]}
              />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Compound Score Visualization */}
      <Card>
        <CardHeader>
          <CardTitle>Compound Scores (Sample)</CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={topPredictionsData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis domain={[-1, 1]} />
              <Tooltip />
              <Bar dataKey="compound" fill="#8884d8">
                {topPredictionsData.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={
                      entry.sentiment === 'positive'
                        ? '#22c55e'
                        : entry.sentiment === 'negative'
                        ? '#ef4444'
                        : '#6b7280'
                    }
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Top Positive Texts */}
      <Card>
        <CardHeader>
          <CardTitle>Top Positive Texts</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          {result.top_positive.slice(0, 5).map((pred, idx) => (
            <div key={idx} className="p-3 border rounded-lg space-y-2">
              <div className="flex items-start justify-between gap-2">
                <p className="text-sm flex-1">{pred.text}</p>
                <Badge variant={getSentimentBadgeVariant(pred.sentiment)}>
                  {pred.sentiment}
                </Badge>
              </div>
              <div className="flex items-center gap-4 text-xs text-muted-foreground">
                <span>Confidence: {pred.confidence.toFixed(3)}</span>
                <span>Compound: {pred.compound.toFixed(3)}</span>
              </div>
              <div className="flex gap-2 text-xs">
                <span className="text-green-600">
                  Pos: {pred.scores.positive.toFixed(2)}
                </span>
                <span className="text-red-600">
                  Neg: {pred.scores.negative.toFixed(2)}
                </span>
                <span className="text-gray-600">
                  Neu: {pred.scores.neutral.toFixed(2)}
                </span>
              </div>
            </div>
          ))}
        </CardContent>
      </Card>

      {/* Top Negative Texts */}
      <Card>
        <CardHeader>
          <CardTitle>Top Negative Texts</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          {result.top_negative.slice(0, 5).map((pred, idx) => (
            <div key={idx} className="p-3 border rounded-lg space-y-2">
              <div className="flex items-start justify-between gap-2">
                <p className="text-sm flex-1">{pred.text}</p>
                <Badge variant={getSentimentBadgeVariant(pred.sentiment)}>
                  {pred.sentiment}
                </Badge>
              </div>
              <div className="flex items-center gap-4 text-xs text-muted-foreground">
                <span>Confidence: {pred.confidence.toFixed(3)}</span>
                <span>Compound: {pred.compound.toFixed(3)}</span>
              </div>
              <div className="flex gap-2 text-xs">
                <span className="text-green-600">
                  Pos: {pred.scores.positive.toFixed(2)}
                </span>
                <span className="text-red-600">
                  Neg: {pred.scores.negative.toFixed(2)}
                </span>
                <span className="text-gray-600">
                  Neu: {pred.scores.neutral.toFixed(2)}
                </span>
              </div>
            </div>
          ))}
        </CardContent>
      </Card>

      {/* All Predictions Table */}
      <Card>
        <CardHeader>
          <CardTitle>All Predictions</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="max-h-96 overflow-y-auto">
            <div className="space-y-2">
              {result.predictions.map((pred, idx) => (
                <div key={idx} className="p-2 border rounded text-xs space-y-1">
                  <div className="flex items-start justify-between gap-2">
                    <p className="flex-1 line-clamp-2">{pred.text}</p>
                    <Badge
                      variant={getSentimentBadgeVariant(pred.sentiment)}
                      className="text-xs"
                    >
                      {pred.sentiment}
                    </Badge>
                  </div>
                  <div className="flex items-center gap-3 text-muted-foreground">
                    <span>Compound: {pred.compound.toFixed(3)}</span>
                    <span>Conf: {pred.confidence.toFixed(2)}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
