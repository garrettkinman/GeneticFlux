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

using GeneticFlux
using Flux.Optimise
using Test
using Random

# TODO: set up for genetic optimizer
@testset "Optimize" begin
  # Ensure rng has different state inside and outside the inner @testset
  # so that w and w' are different
  Random.seed!(84)
  w = randn(10, 10)
  @testset for opt in [Genetic()]
    Random.seed!(42)
    w′ = randn(10, 10)
    b = false
    loss(x) = Flux.Losses.mse(w*x, w′*x .+ b)
    for t = 1: 10^5
      θ = params([w′, b])
      x = rand(10)
      θ̄ = evolve(() -> loss(x), θ)
      Optimise.update!(opt, θ, θ̄)
    end
    @test loss(rand(10, 10)) < 0.01
  end
end