#include <iostream>
#include <fstream>
#include <string>
#include "yourSolution.h"

using namespace lbcrypto;

int main(int argc, char *argv[])
{
    std::string pubKeyLocation;
    std::string multKeyLocation;
    std::string rotKeyLocation;
    std::string ccLocation;
    std::string inputLocation;
    std::string outputLocation;

    for (auto i = 1; i < argc; i += 2)
    {
        std::string arg = argv[i];
        if (arg == "--key_pub" || arg == "--key_public")
        {
            pubKeyLocation = argv[i + 1];
        }
        else if (arg == "--key_mult")
        {
            multKeyLocation = argv[i + 1];
        }
        else if (arg == "--key_rot")
        {
            rotKeyLocation = argv[i + 1];
        }
        else if (arg == "--cc")
        {
            ccLocation = argv[i + 1];
        }
        else if (arg == "--sample")
        {
            inputLocation = argv[i + 1];
        }
        else if (arg == "--output")
        {
            outputLocation = argv[i + 1];
        }
    }

    CKKSTaskSolver CKKSTaskSolver(ccLocation, pubKeyLocation, multKeyLocation, rotKeyLocation, inputLocation,
                                  outputLocation);
    CKKSTaskSolver.eval();
    CKKSTaskSolver.serializeOutput();
    return 0;
}