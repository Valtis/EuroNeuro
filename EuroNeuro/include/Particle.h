#pragma once
#include <vector>
#include <random>

typedef std::vector<double> Array1D;
typedef std::vector<Array1D> Array2D;

class Particle
{
public:
  Particle(int inputs, int hiddens, int outputs);
  void InitializeToRandomValues();
  void UpdateValues(const Particle &bestSolution);
private:
  friend class NeuralNetwork;
  double RandomValue();
  void UpdateValues(double &velocity, double &currentValue, const double &localBest, const double &globalBest);
  
  Array1D m_inputHiddenBiasesCandidates;
  Array1D m_hiddenOutputBiasesCandidates;

  Array2D m_inputHiddenWeightsCandidates;
  Array2D m_hiddenOutputWeightsCandidates;


  Array1D m_inputHiddenBiasesBestCandidates;
  Array1D m_hiddenOutputBiasesBestCandidates;

  Array2D m_inputHiddenWeightsBestCandidates;
  Array2D m_hiddenOutputWeightsBestCandidates;


  Array1D m_inputHiddenBiasesVelocities;
  Array1D m_hiddenOutputBiasesVelocities;

  Array2D m_inputHiddenWeightsVelocities;
  Array2D m_hiddenOutputWeightsVelocities;

  int m_inputNodes;
  int m_hiddenNodes;
  int m_outputNodes;

  double m_bestResult;
  int m_calls;
  std::mt19937 engine;
  std::uniform_real_distribution<double> random;
};
