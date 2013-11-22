#include "FileReader.h"
#include "NeuralNetwork.h"

#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
void NormalizeFields(std::vector<std::vector<double>> &data)
{
  double min[6];
  double max[6];
  for (int i = 0; i < 6; ++i)
  {
    min[i] = 9999999;
    max[i] = -9999999;
  }
  int offset = 0;
  
  if (data[0].size() == 7)
  {
    offset = 1; // ignore first field which is normalized anyway
  }

  for (auto &passenger : data)
  {
    for (int i = 0 + offset; i < 6 + offset; ++i)
    {
      min[i-offset] = std::min(min[i-offset], passenger[i]);
      max[i-offset] = std::max(max[i-offset], passenger[i]);
    }
  }

  for (auto &passenger : data)
  {
    for (int i = 0 + offset; i < 6 + offset; ++i)
    {
      passenger[i] = (passenger[i] - min[i-offset])/(max[i-offset] - min[i-offset]);
    }
  }
}

void CreateLearningAndVerificationData(const std::vector<std::vector<double>> &allData, std::vector<std::vector<double>> &learningData, std::vector<std::vector<double>> &verificationData)
{
  learningData.clear();
  verificationData.clear();

  for (const auto &data : allData)
  {
    if (rand() % 1000 < 200)
    {
      verificationData.push_back(data);
    }
    else
    {
      learningData.push_back(data);
    }
  }
}

void VerifyModel(const std::vector<std::vector<double>> &data)
{
  double result = 0;
  int verificationSetSize = 0;

  for (int k = 0; k < 1; ++k)
  {
    printf("Verification step: %i\n", k);
    std::vector<std::vector<double>> learningData;
    std::vector<std::vector<double>> verificationData;
    CreateLearningAndVerificationData(data, learningData, verificationData);
    verificationSetSize += verificationData.size();
    NeuralNetwork skynet(6, 20, 1);
    
    skynet.Learn(learningData);
    for (const auto &passenger : verificationData)
    {
      std::vector<double> input;
      for (int i = 1; i < passenger.size(); ++i)
      {
        input.push_back(passenger[i]);
      }
      std::vector<double> ret = skynet.Classify(input);
      int value = ret[0] < 0.5 ? 0 : 1;
      printf("Returned: %i Expected: %f\n", value, passenger[0] );
      if (value == passenger[0])
      {
        ++result;
      }
    }
    double temp = result / verificationSetSize;

    temp *= 100;
    printf("Verification set size: %i\n", verificationSetSize);
    printf("Results: %f\n", result);
    printf("Accuracy: %f\n", temp);
  }
}

int main()
{
  srand(time(nullptr));
  setlocale(LC_ALL, "");
	auto data = ReadData("training_set.csv", true);
  auto testData = ReadData("test_set.csv", false);

  setlocale(LC_ALL, "C");

  NormalizeFields(data);
  VerifyModel(data);
 /* NeuralNetwork skynet(6, 20, 1);
  skynet.Learn(data);

  std::ofstream out("out.txt");

  for (auto p : testData)
  {
    auto result = skynet.Classify(p);
    if (result[0] < 0.5)
    {
      out << "0\n";
    }
    else
    {
      out << "1\n";
    }
  }
  */


  system("pause");
}

