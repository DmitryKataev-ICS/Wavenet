namespace Wavenet

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
            |> (*) 1.5
            |> int
        //weights, etc.
        let curMat =
            ref (InnerMatrices
                (
                    DenseMatrix.random<float> indim hiddim (ContinuousUniform(0.0, 1.0)),
                    DenseMatrix.random<float> hiddim outdim (ContinuousUniform(0.0, 1.0)),
                    DenseMatrix.random<float> 1 outdim (ContinuousUniform(0.0, 1.0)),
                    DenseMatrix.random<float> 1 hiddim (ContinuousUniform(fst translation, snd translation)),
                    DenseMatrix.random<float> 1 hiddim (ContinuousUniform(fst dilation, snd dilation))    
                ))
        let oldMat =
            ref (InnerMatrices
                (
                    DenseMatrix.zero indim hiddim,
                    DenseMatrix.zero hiddim outdim,
                    DenseMatrix.zero 1 outdim,
                    DenseMatrix.zero 1 hiddim,
                    DenseMatrix.zero 1 hiddim
                ))
        let mother = motherfunctions.[(int)MotherFunction.MexicanHat]
        let derivative = derivatives.[(int)MotherFunction.MexicanHat]

        //methods
        member x.OldMat with get()=oldMat
        member x.CurMat with get()=curMat
        member x.OutDim with get() = outdim
        member x.Forward(input : Matrix<float>) = 
            let tmp = (input * x.CurMat.Value.InCon.Value - x.CurMat.Value.Trans.Value) ./ x.CurMat.Value.Dil.Value //tmp - row-vector
            tmp.MapInplace (System.Func<float, float>(mother))
            tmp * x.CurMat.Value.OutCon.Value + x.CurMat.Value.Sum.Value //result - row-vector
        
        member x.Backup dChi dM dOmega dT dLambda = 
            let step (x : Matrix<float>) (y : Matrix<float>) (o : Matrix<float>) = x + y.Multiply(etta) + (x - o).Multiply(etta)
            let newMat =
                InnerMatrices
                    (
                        step curMat.Value.InCon.Value dOmega oldMat.Value.InCon.Value,
                        step curMat.Value.OutCon.Value dM oldMat.Value.OutCon.Value,
                        step curMat.Value.Sum.Value dChi oldMat.Value.Sum.Value,
                        step curMat.Value.Trans.Value dT oldMat.Value.Trans.Value,
                        step curMat.Value.Dil.Value dLambda oldMat.Value.Dil.Value
                    )
            x.OldMat := curMat.Value
//            x.OldMat.InCon := curMat.InCon.Value
//            x.OldMat.OutCon := curMat.OutCon.Value
//            x.OldMat.Sum := curMat.Sum.Value
//            x.OldMat.Trans := curMat.Trans.Value
//            x.OldMat.Dil := curMat.Dil.Value
            x.CurMat := newMat
//            x.CurMat.InCon := newMat.InCon.Value
//            x.CurMat.OutCon := newMat.OutCon.Value
//            x.CurMat.Sum := newMat.Sum.Value
//            x.CurMat.Trans := newMat.Trans.Value
//            x.CurMat.Dil := newMat.Dil.Value

            //printfn "%f" (x.CurMat.OutCon.Value.Item(0,0))

        member x.Backward (error : Matrix<float>) (input : Matrix<float>) = 
            //printfn "%f" (error.Item(0,0))
            let protoZ = (input * x.CurMat.Value.InCon.Value - x.CurMat.Value.Trans.Value) ./ curMat.Value.Dil.Value //vector-row
            let Z = protoZ.Map (System.Func<float, float>(mother))
            let Zs = protoZ.Map (System.Func<float, float>(derivative))
            let dChi = error
            let dM = Z.TransposeThisAndMultiply error
            let dOmega = 
                input.TransposeThisAndMultiply(
                    (x.CurMat.Value.OutCon.Value.TransposeAndMultiply error).Transpose() .* (Zs ./ x.CurMat.Value.Dil.Value))
            let dT = error * x.CurMat.Value.OutCon.Value.Transpose() .* (Zs ./ x.CurMat.Value.Dil.Value)
            let dLambda = Zs .* ( (input * x.CurMat.Value.InCon.Value - x.CurMat.Value.Trans.Value) ./ (x.CurMat.Value.Dil.Value .* x.CurMat.Value.Dil.Value) )
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

        let rec subtrain epochs (training : Matrix<float>*Matrix<float>) (validation : Matrix<float>*Matrix<float>) (net : Wavelon) =
            
        //for i in 1..epochs do
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
            printfn "%i\t%f" epochs (local_MSE.RowSums().Sum())
            track := local_MSE :: !track
            if (local_MSE.RowSums().Sum() > 0.001) && (epochs > 0) then
                subtrain (epochs - 1) training validation net
            else (track.Value, net)
        subtrain epochs training validation net