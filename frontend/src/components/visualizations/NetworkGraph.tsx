import * as React from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, Text } from "@react-three/drei";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export interface NetworkLayer {
  name: string;
  nodes: number;
}

export interface NetworkGraphProps {
  layers: NetworkLayer[];
  weights?: number[][][]; // [layer][from_node][to_node]
  title?: string;
  className?: string;
  height?: number;
  showWeights?: boolean;
}

interface NodePosition {
  x: number;
  y: number;
  z: number;
  layer: number;
  index: number;
}

function NetworkVisualization({
  layers,
  weights,
  showWeights = false,
}: {
  layers: NetworkLayer[];
  weights?: number[][][];
  showWeights: boolean;
}) {
  const nodePositions: NodePosition[] = React.useMemo(() => {
    const positions: NodePosition[] = [];
    const layerSpacing = 3;
    const totalWidth = (layers.length - 1) * layerSpacing;

    layers.forEach((layer, layerIndex) => {
      const nodeSpacing = Math.min(1.5, 10 / layer.nodes);
      const totalHeight = (layer.nodes - 1) * nodeSpacing;

      for (let nodeIndex = 0; nodeIndex < layer.nodes; nodeIndex++) {
        positions.push({
          x: layerIndex * layerSpacing - totalWidth / 2,
          y: nodeIndex * nodeSpacing - totalHeight / 2,
          z: 0,
          layer: layerIndex,
          index: nodeIndex,
        });
      }
    });

    return positions;
  }, [layers]);

  const connections = React.useMemo(() => {
    const conns: Array<{
      from: NodePosition;
      to: NodePosition;
      weight?: number;
    }> = [];

    for (let i = 0; i < layers.length - 1; i++) {
      const fromNodes = nodePositions.filter((n) => n.layer === i);
      const toNodes = nodePositions.filter((n) => n.layer === i + 1);

      fromNodes.forEach((from) => {
        toNodes.forEach((to) => {
          const weight = weights?.[i]?.[from.index]?.[to.index];
          conns.push({ from, to, weight });
        });
      });
    }

    return conns;
  }, [layers, nodePositions, weights]);

  return (
    <>
      <ambientLight intensity={0.5} />
      <pointLight position={[10, 10, 10]} />

      {/* Connections */}
      {connections.map((conn, index) => {
        const opacity = showWeights && conn.weight !== undefined
          ? Math.abs(conn.weight)
          : 0.3;
        const color = showWeights && conn.weight !== undefined
          ? conn.weight > 0 ? "#4ade80" : "#f87171"
          : "#888888";

        return (
          <line key={index}>
            <bufferGeometry>
              <bufferAttribute
                attach="attributes-position"
                count={2}
                array={
                  new Float32Array([
                    conn.from.x,
                    conn.from.y,
                    conn.from.z,
                    conn.to.x,
                    conn.to.y,
                    conn.to.z,
                  ])
                }
                itemSize={3}
              />
            </bufferGeometry>
            <lineBasicMaterial color={color} opacity={opacity} transparent />
          </line>
        );
      })}

      {/* Nodes */}
      {nodePositions.map((pos, index) => (
        <mesh key={index} position={[pos.x, pos.y, pos.z]}>
          <sphereGeometry args={[0.2, 32, 32]} />
          <meshStandardMaterial color="#8884d8" />
        </mesh>
      ))}

      {/* Layer labels */}
      {layers.map((layer, index) => {
        const layerNodes = nodePositions.filter((n) => n.layer === index);
        const avgY =
          layerNodes.reduce((sum, n) => sum + n.y, 0) / layerNodes.length;
        const x = layerNodes[0]?.x || 0;

        return (
          <Text
            key={index}
            position={[x, avgY - 2, 0]}
            fontSize={0.3}
            color="#666666"
            anchorX="center"
            anchorY="middle"
          >
            {layer.name}
          </Text>
        );
      })}

      <OrbitControls enableZoom={true} enablePan={true} enableRotate={true} />
    </>
  );
}

export function NetworkGraph({
  layers,
  weights,
  title = "Neural Network",
  className,
  height = 500,
  showWeights = false,
}: NetworkGraphProps) {
  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <div style={{ height }} className="w-full bg-muted/20 rounded">
          <Canvas camera={{ position: [0, 0, 10], fov: 50 }}>
            <NetworkVisualization
              layers={layers}
              weights={weights}
              showWeights={showWeights}
            />
          </Canvas>
        </div>
        <div className="mt-4 text-xs text-muted-foreground space-y-1">
          <p>Use mouse to rotate, zoom, and pan the network</p>
          {showWeights && (
            <p className="flex items-center gap-4">
              <span className="flex items-center gap-1">
                <span className="w-3 h-0.5 bg-green-500" /> Positive weights
              </span>
              <span className="flex items-center gap-1">
                <span className="w-3 h-0.5 bg-red-500" /> Negative weights
              </span>
            </p>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
