## imports

using Flux
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold, logitcrossentropy
using Parameters: @with_kw
using MLDatasets
using MLDatasets: MNIST
using Zygote
using CUDA
using Pipe
using ProgressLogging
using BenchmarkTools
using InteractiveUtils

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
        Dense(32, nclasses)
    )
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
@with_kw mutable struct Genetic <: Flux.Optimise.AbstractOptimiser
    popsize::Int32 = 64
    ϵ::Int32 = 1
end

function apply!(o::Genetic, x, Δ)
    # return a Δ for each element of x (which is a subset of model params)
    # the calling funtion will call x .-= apply!(opt, x, x̄r), where x̄r is just the gradients corresponding to x
    # treat Δ as all zeros to start with, as we will zero-initialize them to feed in,
    # because this package was designed with gradient-based methods in mind, and thus requires you feed in gradients (even if not real)

    # TODO
    # randomly generate a population of Δ candidates (of size o.popsize)
    # perform a feed-through the network for each candidate Δ
    # select the surviving Δ based on minimized loss

end


"""
Functions like Zygote.gradient and returns a delta for how much each given param should change.
Unlike Zygote.gradient, it doesn't calculate any gradients. Instead, it generates a population of mutations,
runs the given training data through each mutated variant of the model, and selects the best performing mutation(s) to return.
"""
function evolve(opt::Genetic, ps::Zygote.Params, f::Function, data)
    # TODO: generate population of mutations
        # store a temporary deep copy of the current params
        # for each mutant
            # randomly mutate a subset of the params
            # run the model over the data
            # if lowest loss of all the mutants so far, keep
        # restore the original params to the model
        # return the best-performing mutant
    # apply each mutation to the model
    # calculate loss for each mutant
end

function train!()
    args = Args()
    model = build_model() |> args.device
    train_data, test_data = getdata(args)

    train_data = train_data .|> args.device
    test_data = test_data .|> args.device

    loss(x,y) = logitcrossentropy(model(x), y)

    ## Training
    evalcb = () -> @show(loss_all(train_data, model))
    opt = ADAM()

    # @epochs args.epochs Flux.train!(loss, params(model), train_data, opt, cb = evalcb)
    ps = Flux.params(model)

    @withprogress for (i, (x, y)) ∈ enumerate(train_data)
        # gradients = gradient(() -> loss(x, y), params)
        mutations = evolve(opt, ps, loss, train_data)
        # println(typeof(gradients))
        # Flux.Optimise.update!(opt, ps, gradients) # <-- need to target this
        Flux.Optimise.update!(opt, ps, mutations)
        @logprogress i / length(enumerate(train_loader))
    end

end

## run

args = Args()
model = build_model()
ps = Flux.params(model)
train_data, test_data = getdata(args)

for (i, (x,y)) in enumerate(train_data)
    @show size.((x, y))
    gradients = gradient(() -> logitcrossentropy(model(x), y), ps)
    @show ps |> length
    for p in ps
        @show p |> size
        @show gradients[p] |> size
        break
    end
    break
end

Zygote.Params
methodswith(Zygote.Grads)
methodswith(Zygote.Params)

train!()

Dense(prod((28,28,1)), 10).weight


train_data, test_data = getdata(Args())

train_data .|> gpu