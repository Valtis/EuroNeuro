#include "NeuralNetwork.h"
#include <cmath>
#include <cstdlib>
#include "Particle.h"

NeuralNetwork::NeuralNetwork(int inputNodes, int hiddenNodes, int outputNodes) : 
  m_inputNodes(inputNodes), m_hiddenNodes(hiddenNodes), m_outputNodes(outputNodes) 
{
  
  m_inputHiddenWeights.resize(inputNodes);
  m_inputHiddenBiases.resize(hiddenNodes);

  for (int i = 0; i < inputNodes; ++i)
  {
    m_inputHiddenWeights[i].resize(hiddenNodes);
  }

  m_hiddenOutputWeights.resize(hiddenNodes);
  m_hiddenOutputBiases.resize(outputNodes);

  for (int i = 0; i < hiddenNodes; ++i)
  {
    m_hiddenOutputWeights[i].resize(outputNodes);
  }
}

NeuralNetwork::~NeuralNetwork()
{

}

double NeuralNetwork::CalculateOutput( const Array1D &inputValues, const Array2D &weights, const Array1D &biases, int node )
{
  double sum = 0;
  for (int i = 0; i < inputValues.size(); ++i)
  {
    sum += inputValues[i] * weights[i][node];
  }

  sum += biases[node];
  if (sum < -45)
  {
    return 0;
  }
  else if (sum > 45)
  {
    return 1;
  }

  return 1.0/(1.0+exp(-sum));
}




std::vector<double> NeuralNetwork::Classify( std::vector<double> &inputs )
{
  std::vector<double> hiddenLayerOutputs;
  std::vector<double> outputLayerOutputs;
  for (int i = 0; i < m_hiddenNodes; ++i)
  {
    hiddenLayerOutputs.push_back( CalculateOutput(inputs, m_inputHiddenWeights, m_inputHiddenBiases, i) );
  }

  for (int i = 0; i < m_outputNodes; ++i)
  {
    outputLayerOutputs.push_back(CalculateOutput(hiddenLayerOutputs, m_hiddenOutputWeights, m_hiddenOutputBiases, i));
  }
  return outputLayerOutputs;
}



void NeuralNetwork::Learn(Array2D learningData)
{
  std::vector<Particle> particles;
  double bestSolutionValue = 1000000000;
  Particle bestSolution(m_inputNodes, m_hiddenNodes, m_outputNodes);

  InitializeParticleSwarm(particles, bestSolution, bestSolutionValue, learningData);

  for (int i = 0; i < 1000; ++i)
  {
    if (i % 100 == 0)
    {
      printf("  Particle swarm optimization step: %i  Lowest error: %lf\n", i, bestSolutionValue);
    }

    for (Particle &p : particles)
    {
      p.UpdateValues(bestSolution);
    }

    for (Particle &p : particles)
    {
      double result = CalculateParticleFitness(p, learningData);
      if (result < bestSolutionValue)
      {
        printf("  Found better solution with error value of: %lf\n", bestSolutionValue);
        bestSolutionValue = result;
        bestSolution = p;
      }

    }
  }


  m_hiddenOutputBiases = bestSolution.m_hiddenOutputBiasesBestCandidates;
  m_inputHiddenBiases = bestSolution.m_inputHiddenBiasesBestCandidates;

  m_inputHiddenWeights = bestSolution.m_inputHiddenWeightsBestCandidates;
  m_hiddenOutputWeights = bestSolution.m_hiddenOutputWeightsBestCandidates ;
  
  printf("\nihWeights:\n");
  for (auto &i : m_inputHiddenWeights)
  {
    for (double &j : i)
    {
      printf("%f ", j);
    }
    printf("\n");
  }

  printf("\nihBiases:\n");
  for (auto &i : m_inputHiddenBiases)
  {
    printf("%f ", i);
  }
   printf("\n");

   printf("\nhoWeights:\n");
   for (auto &i : m_hiddenOutputWeights)
   {
     for (double &j : i)
     {
       printf("%f ", j);
     }
     printf("\n");
   }

   printf("\nhoBiases:\n");
   for (auto &i : m_hiddenOutputBiases)
   {
     printf("%f ", i);
   }
   printf("\n");


}

void NeuralNetwork::InitializeParticleSwarm( std::vector<Particle> &particles, Particle &bestSolution, double &bestSolutionValue, const Array2D &learningData )
{
  for (int i = 0; i < 20; ++i)
  {
    Particle p(m_inputNodes, m_hiddenNodes, m_outputNodes);
    p.InitializeToRandomValues();
    double result = CalculateParticleFitness(p, learningData);
    
    if (result < bestSolutionValue)
    {
      bestSolutionValue = result;
      bestSolution = p;
    }

    particles.push_back(p);
  }
}


double NeuralNetwork::CalculateParticleFitness( Particle &p, const Array2D &learningData )
{
  double results = 0;
  for (const auto &data : learningData)
  {
    std::vector<double> hiddenLayerOutputs;
    std::vector<double> outputLayerOutputs;
    Array1D inputs;
    for (int i = 1; i < data.size(); ++i)
    {
      inputs.push_back(data[i]);
    }
    for (int i = 0; i < m_hiddenNodes; ++i)
    {
      hiddenLayerOutputs.push_back( CalculateOutput(inputs, p.m_inputHiddenWeightsCandidates, p.m_inputHiddenBiasesCandidates, i) );
    }

    for (int i = 0; i < m_outputNodes; ++i)
    {
      outputLayerOutputs.push_back(CalculateOutput(hiddenLayerOutputs, p.m_hiddenOutputWeightsCandidates, p.m_hiddenOutputBiasesCandidates, i));
    }

    double temp = outputLayerOutputs[0] - data[0];
    results += temp*temp;
    
  }

  results /= learningData.size();

  if (results < p.m_bestResult)
  {
    p.m_bestResult = results;

    p.m_inputHiddenWeightsBestCandidates = p.m_inputHiddenWeightsCandidates;
    p.m_inputHiddenBiasesBestCandidates = p.m_inputHiddenBiasesCandidates;

    p.m_hiddenOutputBiasesBestCandidates = p.m_hiddenOutputBiasesCandidates;
    p.m_hiddenOutputWeightsBestCandidates = p.m_hiddenOutputWeightsCandidates;
  }

  return results;

}
