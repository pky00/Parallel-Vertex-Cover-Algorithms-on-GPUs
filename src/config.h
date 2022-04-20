
#ifndef CONFIG_H
#define CONFIG_H

#include <unistd.h>
#include <stdio.h>
#include <cstring>
#include <cstdlib>

#ifndef USE_COUNTERS
#define USE_COUNTERS 0
#endif

static void usage()
{
    fprintf(stderr,
            "\n"
            "Arguments :\n"
            "\n"
            "   -v <Version>                        Used to set the version of the code to run.\n"
            "                                  - Options : hybrid, stackonly, sequential\n"
            "                                  - Default : hybrid\n"
            "\n"
            "   -f <Graph File Name>                Used to pass file name containing the graph.\n"
            "                                  - Default : GraphInput.txt\n"
            "\n"
            "   -o <Output File Prefix>             Used to pass the output file prefix \n"
            "                                  (unique identifier) to be used in the Results.csv \n"
            "                                  directory and the filenames of the data files \n"
            "                                  outputed in Counters/ and NODES_PER_SM/ .\n"
            "                                  - Default : 1\n"
            "\n"
            "   -i <Instance>                       Used to set the instance, minimum vertex cover\n"
            "                                  or parameterized vertex cover, the code will run.\n"
            "                                  - Options : mvc, pvc\n"
            "                                  - Default : mvc\n"
            "\n"
            "   -k <paramter>                       Used to set parameter on which parameterized \n"
            "                                  vertex cover is run on. Should be and Integer.\n"
            "                                  If you are not running parameterized vertex cover\n"
            "                                  this argument won't have any effect.\n"
            "                                  - Default : 0\n"
            "\n"
            "   -q <Global Work List Size>          Used to set the global work list size. Must\n"
            "                                  be and Integer and a power of 2\n"
            "                                  - Default : 2^19\n"
            "\n"
            "   -t <Global Work List Threashold>    Used to set the global work list threashold.\n"
            "                                  Must be a Float between 0.0 and 1.0 . It represent\n"
            "                                  a percentage of the global work list.\n"
            "                                  - Default : 1.0\n"
            "\n"
            "   -d <Starting Depth>                 Used to set the starting depth for the \n"
            "                                  stackonly version. Must be an Integer. If you \n"
            "                                  are not running the stackonly version this \n"
            "                                  argument won't have any effect.\n"
            "                                  - Default : 4\n"
            "\n"
            "   -b <Block Dimension>                Used to set the block dimension to be used\n"
            "                                  by the GPU. Must be an Integer. If not set by \n"
            "                                  the user it will be determined by the code \n"
            "                                  according to the abilities of the gpu and the\n"
            "                                  size of the graph.\n"
            "\n"
            "   -g <Use Global Memory>              Used to determine if the code should use \n"
            "                                  global memory for the vertex degree arrays\n"
            "                                  (Mentioned in Referenced Paper). Is either 0 \n"
            "                                  or 1. If not provided by the user it will be \n"
            "                                  determined by the code according to the \n"
            "                                  abilities of the gpu and the size of the graph.\n"
            "\n"
            "   -n <Number of Blocks>               Used to determine the numer of blocks for \n"
            "                                  the gpu to run. Must be an Integer. If not \n"
            "                                  determined by the user it will be determined \n"
            "                                  by the code according to the abilities of the\n"
            "                                  gpu and the size of the graph.\n"
            "\n"
            "\n");
}

enum Version
{
    STACK_ONLY,
    HYBRID,
    SEQUENTIAL,
};

static Version parseVersion(const char *s)
{
    if (strcmp(s, "stackonly") == 0)
    {
        return STACK_ONLY;
    }
    else if (strcmp(s, "hybrid") == 0)
    {
        return HYBRID;
    }
    else if (strcmp(s, "sequential") == 0)
    {
        return SEQUENTIAL;
    }
    else
    {
        fprintf(stderr, "Unrecognized -v option: %s\n", s);
        exit(0);
    }
}

static const char *asString(Version version)
{
    switch (version)
    {
    case STACK_ONLY:
        return "stackonly";
    case HYBRID:
        return "hybrid";
    case SEQUENTIAL:
        return "sequential";
    default:
        fprintf(stderr, "Unrecognized version\n");
        exit(0);
    }
}

enum Instance
{
    MVC,
    PVC,
};

static Instance parseInstance(const char *s)
{
    if (strcmp(s, "mvc") == 0)
    {
        return MVC;
    }
    else if (strcmp(s, "pvc") == 0)
    {
        return PVC;
    }
    else
    {
        fprintf(stderr, "Unrecognized -i option: %s\n", s);
        exit(0);
    }
}

static const char *asString(Instance instance)
{
    switch (instance)
    {
    case MVC:
        return "mvc";
    case PVC:
        return "pvc";
    default:
        fprintf(stderr, "Unrecognized instance\n");
        exit(0);
    }
}

struct Config
{
    Version version;
    Instance instance;
    unsigned int k;
    unsigned int numBlocks;
    bool userDefMemory;
    bool useGlobalMemory;
    unsigned int blockDim;
    unsigned int globalListSize;
    float globalListThreshold;
    unsigned int startingDepth;
    const char *outputFilePrefix;
    const char *graphFileName;
};

static Config parseArgs(int argc, char **argv)
{
    Config config;
    config.version = HYBRID;
    config.instance = MVC;
    config.k = 0;
    config.blockDim = 0;
    config.numBlocks = 0;
    config.userDefMemory = false;
    config.useGlobalMemory = false;
    config.globalListSize = 1 << 19;
    config.globalListThreshold = 1.0;
    config.startingDepth = 4;
    config.outputFilePrefix = "1";
    config.graphFileName = "GraphInput.txt";

    int opt;
    while ((opt = getopt(argc, argv, "v:f:o:i:k:q:t:d:b:g:n:h")) >= 0)
    {
        switch (opt)
        {
        case 'v':
            config.version = parseVersion(optarg);
            break;
        case 'f':
            config.graphFileName = optarg;
            break;
        case 'o':
            config.outputFilePrefix = optarg;
            break;
        case 'i':
            config.instance = parseInstance(optarg);
            break;
        case 'k':
            config.k = atoi(optarg);
            break;
        case 'q':
            config.globalListSize = atoi(optarg);
            break;
        case 't':
            config.globalListThreshold = atof(optarg);
            break;
        case 'd':
            config.startingDepth = atoi(optarg);
            break;
        case 'b':
            config.blockDim = atoi(optarg);
            break;
        case 'g':
            config.userDefMemory = true;
            config.useGlobalMemory = atoi(optarg);
            break;
        case 'n':
            config.numBlocks = atoi(optarg);
            break;
        case 'h':
            usage();
            exit(0);
        default:
            fprintf(stderr, "\nUnrecognized option!\n");
            exit(0);
        }
    }

    return config;
}

#endif
