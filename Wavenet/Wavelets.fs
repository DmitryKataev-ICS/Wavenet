namespace Wavenet

module Wavelets =

    type MotherFunction=
        | Morlet = 0
        | MexicanHat = 1

    let motherfunctions = 
        [|
            fun x -> 0.75112554446494 * cos(x * 5.336446256636997) * exp((-(x * x) / 2.0));
            fun x -> (1.0 - (x * x)) * exp((-(x * x)) / 2.0)
        |]

    let derivatives = 
        [|
            fun x -> -4.008341100024355 * exp((-(x * x)) / 2.0) * sin(5.336446256636997 * x) - 0.75112554446494 * x * exp((-(x * x)) / 2.0) * cos(5.336446256636997 * x);
            fun x -> -x * (1.0 - (x * x)) * exp(-(x * x) / 2.0) - 2.0 * x * exp((-(x * x)) / 2.0)
        |]

    let from_freq =
        [|
            fun freq -> 0.8458 / freq + 0.0005407;
            fun freq -> abs(0.2282 / freq - 0.001325)
        |]