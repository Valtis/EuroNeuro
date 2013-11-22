#include "FileReader.h"
#include <cstring>
#include <string>
#include <fstream>
#pragma warning ( disable : 4996 )
std::vector<std::string> Tokenize(std::string text, std::string delimiters)
{
  std::vector<std::string> tokens;
  if (text.size() == 0)
  {
    return tokens;
  }

  char *retVal;
  const unsigned int textbufferSize = text.length() + 1;
  char *textBuffer = new char[textbufferSize];
  for (unsigned int i = 0; i < textbufferSize; ++i)
  {
    textBuffer[i] = '\0';
  }

  strncpy(textBuffer, text.c_str(), text.length());
  retVal = strtok(textBuffer, delimiters.c_str());


  while (retVal != nullptr)
  {
    tokens.push_back(retVal);
    retVal = strtok(nullptr, delimiters.c_str());
  }

  delete [] textBuffer;
  return tokens;
}

std::vector<std::vector<double>> ReadData(const char *fileName, bool isLearningData)
{
  unsigned int expectedTokenCount = 7;
  if (!isLearningData)
  {
    --expectedTokenCount;
  }
 
  std::ifstream file(fileName);
  if (!file.is_open())
  {
    throw std::runtime_error("Couldn't find training_set.csv");
  }
  std::string line;

  std::vector<std::vector<double>> passengerData;
  while (getline(file, line))
  {
    std::vector<double> passenger;
    auto tokens = Tokenize(line, ";");


    int pos = 0;
    if (isLearningData)
    {
      passenger.push_back( atoi(tokens[pos++].c_str()));
    }
    passenger.push_back(atoi(tokens[pos++].c_str()));
    passenger.push_back(!strcmp("female", tokens[pos++].c_str()) ? 0 : 1);

    if (tokens.size() == expectedTokenCount)
    {
      passenger.push_back(atoi(tokens[pos++].c_str()));
    }
    else
    {
      passenger.push_back(40); // somewhere around half of  average life span 
    }

    passenger.push_back(atoi(tokens[pos++].c_str()));
    passenger.push_back(atoi(tokens[pos++].c_str()));
    passenger.push_back(atof(tokens[pos++].c_str()));
    passengerData.push_back(passenger);
  }

  return passengerData;
}