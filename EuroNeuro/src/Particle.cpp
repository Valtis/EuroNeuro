#include "Particle.h"
#include <ctime>
#define LIMIT 45



Particle::Particle(int inputNodes, int hiddenNodes, int outputNodes) : 
  m_inputNodes(inputNodes), m_hiddenNodes(hiddenNodes), m_outputNodes(outputNodes), m_calls(0), m_bestResult(10000000),
  random(-LIMIT, LIMIT)
{
  engine.seed(time(nullptr));

  m_inputHiddenWeightsCandidates.resize(inputNodes);
  m_inputHiddenWeightsBestCandidates.resize(inputNodes);
  m_inputHiddenWeightsVelocities.resize(inputNodes);

  m_inputHiddenBiasesCandidates.resize(hiddenNodes);
  m_inputHiddenBiasesBestCandidates.resize(hiddenNodes);
  m_inputHiddenBiasesVelocities.resize(hiddenNodes);

  for (int i = 0; i < inputNodes; ++i)
  {
    m_inputHiddenWeightsCandidates[i].resize(hiddenNodes);
    m_inputHiddenWeightsVelocities[i].resize(hiddenNodes);
    m_inputHiddenWeightsBestCandidates[i].resize(hiddenNodes);
  }

  m_hiddenOutputWeightsCandidates.resize(hiddenNodes);
  m_hiddenOutputWeightsVelocities.resize(hiddenNodes);
  m_hiddenOutputWeightsBestCandidates.resize(hiddenNodes);

  m_hiddenOutputBiasesCandidates.resize(outputNodes);
  m_hiddenOutputBiasesVelocities.resize(outputNodes);
  m_hiddenOutputBiasesBestCandidates.resize(outputNodes);

  for (int i = 0; i < hiddenNodes; ++i)
  {
    m_hiddenOutputWeightsCandidates[i].resize(outputNodes);
    m_hiddenOutputWeightsVelocities[i].resize(outputNodes);
    m_hiddenOutputWeightsBestCandidates[i].resize(outputNodes);
  }
}


double Particle::RandomValue()
{
  return random(engine);
}

void Particle::InitializeToRandomValues()
{
  for (int i = 0; i < m_hiddenNodes; ++i)
  {
    for (int j = 0; j < m_inputNodes; ++j)
    {
      m_inputHiddenWeightsCandidates[j][i] = RandomValue();
      m_inputHiddenWeightsBestCandidates[j][i] = m_inputHiddenWeightsCandidates[j][i];
      m_inputHiddenWeightsVelocities[j][i] = 0;
    }

    m_inputHiddenBiasesCandidates[i] = RandomValue();
    m_inputHiddenBiasesBestCandidates[i] = m_inputHiddenBiasesCandidates[i];
    m_inputHiddenBiasesVelocities[i] = 0;
  }

  for (int i = 0; i < m_outputNodes; ++i)
  {
    for (int j = 0; j < m_hiddenNodes; ++j)
    {
      m_hiddenOutputWeightsCandidates[j][i] = RandomValue();
      m_hiddenOutputWeightsBestCandidates[j][i] = m_hiddenOutputWeightsCandidates[j][i];
      m_hiddenOutputWeightsVelocities[j][i] = 0;
    }

    m_hiddenOutputBiasesCandidates[i] = RandomValue();
    m_hiddenOutputBiasesBestCandidates[i] = m_hiddenOutputBiasesCandidates[i];
    m_hiddenOutputBiasesVelocities[i]= 0;
  }
}

void Particle::UpdateValues(double &velocity, double &currentValue, const double &localBest, const double &globalBest)
{
  std::uniform_real_distribution<double> r(0, 1);
  double localSolutionMultiplier = r(engine);
  double globalSolutionMultiplier = r(engine);

  const double velocityMultiplier = 0.729;
  const double localVelocityMultiplier = 1.49445;
  const double globalVelocityMultiplier = 1.49445;

  velocity = velocity*velocityMultiplier + 
    localSolutionMultiplier*localVelocityMultiplier*(localBest - currentValue) +
    globalSolutionMultiplier*globalVelocityMultiplier*(globalBest - currentValue);

  if (velocity < -LIMIT)
  {
    velocity = -LIMIT;
  } 
  else if (velocity > LIMIT)
  {
    velocity = LIMIT;
  }

  currentValue += velocity;

  if (currentValue > LIMIT)
  {
    currentValue = LIMIT;
  } 
  else if (currentValue < - LIMIT)
  {
    currentValue = -LIMIT;
  }

}

void Particle::UpdateValues(const Particle &bestSolution)
{
  ++m_calls;

  for (int i = 0; i < m_hiddenNodes; ++i)
  {
    for (int j = 0; j < m_inputNodes; ++j)
    {
      UpdateValues(m_inputHiddenWeightsVelocities[j][i], 
        m_inputHiddenWeightsCandidates[j][i], 
        m_inputHiddenWeightsBestCandidates[j][i], 
        bestSolution.m_inputHiddenWeightsBestCandidates[j][i]);

    }
    UpdateValues(m_inputHiddenBiasesVelocities[i], 
      m_inputHiddenBiasesCandidates[i], 
      m_inputHiddenBiasesBestCandidates[i], 
      bestSolution.m_inputHiddenBiasesBestCandidates[i]);
  }

  for (int i = 0; i < m_outputNodes; ++i)
  {
    for (int j = 0; j < m_hiddenNodes; ++j)
    {
      UpdateValues(m_hiddenOutputWeightsVelocities[j][i], 
        m_hiddenOutputWeightsCandidates[j][i], 
        m_hiddenOutputWeightsBestCandidates[j][i], 
        bestSolution.m_hiddenOutputWeightsBestCandidates[j][i]);
    }
    UpdateValues(m_hiddenOutputBiasesVelocities[i], 
      m_hiddenOutputBiasesCandidates[i], 
      m_hiddenOutputBiasesBestCandidates[i], 
      bestSolution.m_hiddenOutputBiasesBestCandidates[i]);

  }


}