# Copyright 2023 Garrett Kinman
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

using Flux
using Zygote: Params

abstract type AbstractEvolver end

# TODO: add crossover functions
"""
    Genetic(popsize = 100, ϵ = 1)

Genetic algorithm-based optimizer with a given population size, crossover rate, mutation rate, and ϵ survivors of each generation.
"""
struct Genetic <: AbstractEvolver
    popsize::Int32
    crossover_rate::Float32
    mutation_rate::Float32
    ε::Int32
    selection::Function
    crossover::Function
    mutation::Function
end

function mutate!(evolver::Genetic, xs::Params)
    
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
Works like Zygote.gradient and returns a delta for how much each given param will change.
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

# TODO: add adaptive genetic optimizer, which modifies the crossover and mutation rates over time