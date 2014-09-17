﻿namespace Wavenet

module Wavelon = 

    open Wavelets
    open MathNet.Numerics
    open MathNet.Numerics.LinearAlgebra
    open MathNet.Numerics.Distributions

    let etta = 0.005

    type InnerMatrices
        (
            inconnections: Matrix<float>,
            outconnections : Matrix<float>,
            summer : Matrix<float>,
            translations : Matrix<float>,
            dilations : Matrix<float>) =
        member x.InCon = ref inconnections
        member x.OutCon = ref outconnections
        member x.Sum = ref summer
        member x.Trans = ref translations
        member x.Dil = ref dilations

    type Wavelon(indim, outdim, ts_length) = 
        
        //magic
        let freq = (0.001, 1.0)
        let dilation = (from_freq.[(int)MotherFunction.MexicanHat] (snd freq), from_freq.[(int)MotherFunction.MexicanHat] (snd freq))
        let translation = (-5.0, 5.0)
        //dimensions
        let indim = indim
        let outdim = outdim
        (*
        Arnold - Kolmogorov - Hecht-Nielsen theorem in minimalistic form
        self.hidden_power = int(
            (
                output_power * education_power 
                / 
                (1 + log(education_power, 2))
            ) 
            / (input_power + output_power))
        *)
        let hiddim = 
            ( float(outdim * ts_length) / (1.0 + (ts_length|>float|>log) / (log 2.0)) ) / ( (indim + outdim) |> float )
            |> int
        //weights, etc.
        let curMat =
            InnerMatrices
                (
                    DenseMatrix.random<float> indim hiddim (ContinuousUniform(0.0, 1.0)),
                    DenseMatrix.random<float> hiddim outdim (ContinuousUniform(0.0, 1.0)),
                    DenseMatrix.random<float> 1 outdim (ContinuousUniform(0.0, 1.0)),
                    DenseMatrix.random<float> 1 hiddim (ContinuousUniform(fst translation, snd translation)),
                    DenseMatrix.random<float> 1 hiddim (ContinuousUniform(fst dilation, snd dilation))    
                )
        let oldMat =
            InnerMatrices
                (
                    DenseMatrix.zero indim hiddim,
                    DenseMatrix.zero hiddim outdim,
                    DenseMatrix.zero 1 outdim,
                    DenseMatrix.zero 1 hiddim,
                    DenseMatrix.zero 1 hiddim
                )
        let mother = motherfunctions.[(int)MotherFunction.MexicanHat]
        let derivative = derivatives.[(int)MotherFunction.MexicanHat]

        //methods
        member x.OutDim with get() = outdim
        member x.Forward(input : Matrix<float>) = 
            let tmp = (input * curMat.InCon.Value - curMat.Trans.Value) ./ curMat.Dil.Value //tmp - row-vector
            tmp.MapInplace (System.Func<float, float>(mother))
            tmp * curMat.OutCon.Value + curMat.Sum.Value //result - row-vector
        
        member x.Backup dChi dM dOmega dT dLambda = 
            let step (x : Matrix<float>) (y : Matrix<float>) (o : Matrix<float>) = x + y.Multiply(etta) + (x - o).Multiply(etta)
            let newMat =
                InnerMatrices
                    (
                        step curMat.InCon.Value dOmega oldMat.InCon.Value,
                        step curMat.OutCon.Value dM oldMat.OutCon.Value,
                        step curMat.Sum.Value dChi oldMat.Sum.Value,
                        step curMat.Trans.Value dT oldMat.Trans.Value,
                        step curMat.Dil.Value dLambda oldMat.Dil.Value
                    )

            oldMat.InCon := curMat.InCon.Value
            oldMat.OutCon := curMat.OutCon.Value
            oldMat.Sum := curMat.Sum.Value
            oldMat.Trans := curMat.Trans.Value
            oldMat.Dil := curMat.Dil.Value

            curMat.InCon := newMat.InCon.Value
            curMat.OutCon := newMat.OutCon.Value
            curMat.Sum := newMat.Sum.Value
            curMat.Trans := newMat.Trans.Value
            curMat.Dil := newMat.Dil.Value

        member x.Backward (error : Matrix<float>) (input : Matrix<float>) = 
            let protoZ = (input * curMat.InCon.Value - curMat.Trans.Value) ./ curMat.Dil.Value //vector-row
            let Z = protoZ.Map (System.Func<float, float>(mother))
            let Zs = protoZ.Map (System.Func<float, float>(derivative))
            let dChi = error
            let dM = Z.TransposeThisAndMultiply error
            let dOmega = 
                input.TransposeThisAndMultiply(
                    (curMat.OutCon.Value.TransposeAndMultiply error).Transpose() .* (Zs ./ curMat.Dil.Value))
            let dT = error * curMat.OutCon.Value.Transpose() .* (Zs ./ curMat.Dil.Value)
            let dLambda = Zs .* ( (input * curMat.InCon.Value - curMat.Trans.Value) ./ (curMat.Dil.Value .* curMat.Dil.Value) )
            x.Backup dChi dM dOmega dT dLambda

    let train epochs (training : Matrix<float>*Matrix<float>) (validation : Matrix<float>*Matrix<float>) (net : Wavelon) =
        let rand = new System.Random()

        let swap (a: _[]) x y =
            let tmp = a.[x]
            a.[x] <- a.[y]
            a.[y] <- tmp

        // shuffle an array (in-place)
        let shuffle a =
            Array.iteri (fun i _ -> swap a i (rand.Next(i, Array.length a))) a

        let track = ref []
        for i in 1..epochs do
            // fst - in; snd - out
            let (loc_in, loc_out) = 
                let tmpin = fst training
                let tmpout = snd training
                let inversion = [|0..(tmpin.RowCount-1)|]
                shuffle inversion
                let permutation = Permutation.FromInversions inversion
                tmpin.PermuteRows permutation
                tmpout.PermuteRows permutation
                (tmpin, tmpout)
            Array.iter 
                (fun i -> net.Backward (((loc_out.Row i).ToRowMatrix() ) - (net.Forward ((loc_in.Row i).ToRowMatrix()))) ((loc_in.Row i).ToRowMatrix()) )
                [|0..(loc_in.RowCount-1)|]
            let local_MSE =
                let errors = 
                    Array.map 
                        (fun i -> ((snd validation).Row i).ToRowMatrix() - (net.Forward (((fst validation).Row i).ToRowMatrix())) ) 
                        [|0..((fst validation).RowCount-1)|]
                let sq_errors = Array.map (fun (m : Matrix<float>) -> Matrix.map (fun x -> x*x) m) errors
                Array.reduce (+) sq_errors |> (*) 0.5
            let aux = (fst validation).RowCount * net.OutDim |> float
            Matrix.mapInPlace (fun x -> x / aux) local_MSE
            track := local_MSE :: !track
        (track.Value, net)