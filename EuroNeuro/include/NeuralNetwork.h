#pragma once
#include <vector>
typedef std::vector<double> Array1D;
typedef std::vector<Array1D> Array2D;

class Particle;
class NeuralNetwork
{
public:
  NeuralNetwork(int inputNodes, int hiddenNodes, int outputNodes);
  ~NeuralNetwork();
  void Learn(Array2D learningData);
  std::vector<double> Classify(Array1D &inputs);
private:
  double CalculateOutput(const Array1D &inputValues, const Array2D &ihWeights, const Array1D &ihBiases, int hiddenNode);
  
  void InitializeToRandomValues();
  int RandomValue();
  void InitializeParticleSwarm(std::vector<Particle> &particles, Particle &bestSolution, double &bestSolutionValue, const Array2D &learningData);
  double CalculateParticleFitness(Particle &p, const Array2D &learningData);
  
  int m_inputNodes;
  int m_hiddenNodes;
  int m_outputNodes;

  Array2D m_inputHiddenWeights;
  Array1D m_inputHiddenBiases;

  Array2D m_hiddenOutputWeights;
  Array1D m_hiddenOutputBiases;
};