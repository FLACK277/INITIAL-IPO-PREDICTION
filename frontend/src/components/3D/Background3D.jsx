import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Stars, Float } from '@react-three/drei';
import { useRef, useMemo } from 'react';
import * as THREE from 'three';

// Floating geometric shape
function FloatingShape({ position, rotation, scale, color }) {
  const meshRef = useRef();
  
  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.x += 0.003;
      meshRef.current.rotation.y += 0.002;
      meshRef.current.position.y = position[1] + Math.sin(state.clock.elapsedTime * 0.5) * 0.2;
    }
  });

  return (
    <Float
      speed={1.5}
      rotationIntensity={0.5}
      floatIntensity={0.5}
      position={position}
    >
      <mesh ref={meshRef} rotation={rotation} scale={scale}>
        <icosahedronGeometry args={[1, 0]} />
        <meshStandardMaterial
          color={color}
          wireframe
          emissive={color}
          emissiveIntensity={0.5}
          transparent
          opacity={0.6}
        />
      </mesh>
    </Float>
  );
}

// Particle field
function ParticleField() {
  const particlesRef = useRef();
  
  const particles = useMemo(() => {
    const temp = [];
    for (let i = 0; i < 200; i++) {
      temp.push({
        position: [
          (Math.random() - 0.5) * 30,
          (Math.random() - 0.5) * 30,
          (Math.random() - 0.5) * 30,
        ],
      });
    }
    return temp;
  }, []);

  useFrame((state) => {
    if (particlesRef.current) {
      particlesRef.current.rotation.y += 0.0002;
    }
  });

  return (
    <group ref={particlesRef}>
      {particles.map((particle, i) => (
        <mesh key={i} position={particle.position}>
          <sphereGeometry args={[0.02, 8, 8]} />
          <meshBasicMaterial color="#00d4ff" transparent opacity={0.6} />
        </mesh>
      ))}
    </group>
  );
}

// Candlestick chart representation
function CandlestickChart({ position }) {
  const groupRef = useRef();
  
  useFrame((state) => {
    if (groupRef.current) {
      groupRef.current.rotation.y = Math.sin(state.clock.elapsedTime * 0.3) * 0.2;
    }
  });

  const candlesticks = useMemo(() => {
    return Array.from({ length: 8 }, (_, i) => ({
      height: 0.5 + Math.random() * 1.5,
      color: Math.random() > 0.5 ? '#00ff88' : '#ff4444',
      x: i * 0.3 - 1.2,
    }));
  }, []);

  return (
    <group ref={groupRef} position={position}>
      {candlesticks.map((candle, i) => (
        <Float key={i} speed={2} rotationIntensity={0} floatIntensity={0.3}>
          <mesh position={[candle.x, candle.height / 2, 0]}>
            <boxGeometry args={[0.15, candle.height, 0.15]} />
            <meshStandardMaterial
              color={candle.color}
              emissive={candle.color}
              emissiveIntensity={0.3}
            />
          </mesh>
        </Float>
      ))}
    </group>
  );
}

// Main 3D Scene
function Scene3D() {
  return (
    <>
      {/* Lighting */}
      <ambientLight intensity={0.3} />
      <pointLight position={[10, 10, 10]} intensity={0.8} color="#00d4ff" />
      <pointLight position={[-10, -10, -10]} intensity={0.5} color="#ffd700" />
      <spotLight
        position={[0, 15, 0]}
        angle={0.3}
        penumbra={1}
        intensity={0.5}
        color="#00ff88"
      />

      {/* Stars background */}
      <Stars
        radius={100}
        depth={50}
        count={5000}
        factor={4}
        saturation={0}
        fade
        speed={1}
      />

      {/* Floating shapes */}
      <FloatingShape
        position={[-5, 2, -5]}
        rotation={[0, 0, 0]}
        scale={0.8}
        color="#00d4ff"
      />
      <FloatingShape
        position={[5, -2, -8]}
        rotation={[0, 0, 0]}
        scale={1}
        color="#00ff88"
      />
      <FloatingShape
        position={[3, 4, -6]}
        rotation={[0, 0, 0]}
        scale={0.6}
        color="#ffd700"
      />

      {/* Candlestick charts */}
      <CandlestickChart position={[-6, -1, -10]} />
      <CandlestickChart position={[6, 3, -12]} />

      {/* Particle field */}
      <ParticleField />

      {/* Orbit controls for subtle camera movement */}
      <OrbitControls
        enableZoom={false}
        enablePan={false}
        autoRotate
        autoRotateSpeed={0.3}
        maxPolarAngle={Math.PI / 2}
        minPolarAngle={Math.PI / 2}
      />
    </>
  );
}

// Main component
export default function Background3D() {
  return (
    <div className="fixed top-0 left-0 w-full h-full -z-10">
      <Canvas
        camera={{ position: [0, 0, 8], fov: 60 }}
        gl={{ antialias: true, alpha: true }}
        style={{ background: 'transparent' }}
      >
        <Scene3D />
      </Canvas>
    </div>
  );
}
