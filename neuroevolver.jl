## imports

using Flux
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold, logitcrossentropy
using Parameters: @with_kw
using MLDatasets: MNIST
using Zygote
using CUDA
using Pipe
using ProgressLogging
using BenchmarkTools

## prepare MNIST data

@with_kw mutable struct Args
    batchsize::Int = 1024   # batch size
    epochs::Int = 10        # number of epochs
    device::Function = gpu  # set as gpu, if gpu available
end

function getdata(args)
    # Loading Dataset	
    xtrain, ytrain = MLDatasets.MNIST.traindata(Float32)
    xtest, ytest = MLDatasets.MNIST.testdata(Float32)
	
    # Reshape Data in order to flatten each image into a linear array
    xtrain = Flux.flatten(xtrain)
    xtest = Flux.flatten(xtest)

    # One-hot-encode the labels
    ytrain, ytest = onehotbatch(ytrain, 0:9), onehotbatch(ytest, 0:9)

    # Batching
    train_data = DataLoader((xtrain, ytrain), batchsize=args.batchsize, shuffle=true)
    test_data = DataLoader((xtest, ytest), batchsize=args.batchsize)

    return train_data, test_data
end

## build model

function build_model(; imgsize=(28,28,1), nclasses=10)
    return Chain(
 	    Dense(prod(imgsize), 32, relu),
            Dense(32, nclasses))
end

function loss_all(dataloader, model)
    l = 0f0
    for (x,y) in dataloader
        l += logitcrossentropy(model(x), y)
    end
    l/length(dataloader)
end

function accuracy(data_loader, model)
    acc = 0
    for (x,y) in data_loader
        acc += sum(onecold(cpu(model(x))) .== onecold(cpu(y)))*1 / size(x,2)
    end
    acc/length(data_loader)
end

"""
    Genetic(popsize = 64, ϵ = 1)

Genetic algorithm-based optimizer with population size for each generation of `popsize`, and `ϵ` survivors of each generation.

"""
mutable struct Genetic <: Flux.Optimise.AbstractOptimiser
    popsize::Int32 = 64
    ϵ::Int32 = 1
end

function apply!(o::Genetic, x, Δ)
    # TODO
  end

function train!()
    args = Args()
    model = build_model() |> args.device
    train_data, test_data = getdata(args)

    train_data = train_data .|> args.device
    test_data = test_data .|> args.device



end

## run

Dense(prod((28,28,1)), 10).weight


train_data, test_data = getdata(Args())

train_data .|> gpu