namespace Wavenet

module Main =

    open Wavelon
    open MathNet.Numerics.Data.Text
    open MathNet.Numerics.LinearAlgebra

    [<EntryPoint>]
    let main argv = 
        let tsi = DelimitedReader.Read<float>("tsi.csv", false, ",", false, new System.Globalization.CultureInfo("en-US"))
        let tso = DelimitedReader.Read<float>("tso.csv", false, ",", false, new System.Globalization.CultureInfo("en-US"))
        let vsi = DelimitedReader.Read<float>("vsi.csv", false, ",", false, new System.Globalization.CultureInfo("en-US"))
        let vso = DelimitedReader.Read<float>("vso.csv", false, ",", false, new System.Globalization.CultureInfo("en-US"))
        let tesi = DelimitedReader.Read<float>("tesi.csv", false, ",", false, new System.Globalization.CultureInfo("en-US"))
        let teso = DelimitedReader.Read<float>("teso.csv", false, ",", false, new System.Globalization.CultureInfo("en-US"))

        let indim = tsi.ColumnCount
        let outdim = tso.ColumnCount
        let tslength = tsi.RowCount

        let net = Wavelon(indim, outdim, tslength)
        let (track, trained_net) = train 200 (tsi, tso) (vsi, vso) net

        let openloop_validation =
            List.map
                (fun i -> net.Forward ((vsi.Row i).ToRowMatrix()))
                [0..(vsi.RowCount-1)]
            |> List.map
                (fun (x : Matrix<float>) -> x.Transpose())
            |> List.reduce
                (fun (a : Matrix<float>) (b : Matrix<float>) -> a.Append b)
        DelimitedWriter.Write( "validation.csv", openloop_validation.Transpose(), ";")
        let openloop_test =
            List.map
                (fun i -> net.Forward ((tesi.Row i).ToRowMatrix()))
                [0..(tesi.RowCount-1)]
            |> List.map
                (fun (x : Matrix<float>) -> x.Transpose())
            |> List.reduce
                (fun (a : Matrix<float>) (b : Matrix<float>) -> a.Append b)
        DelimitedWriter.Write( "test.csv", openloop_test.Transpose(), ";")
        //gazp.PermuteColumns
        //MathNet.Numerics.Permutation
        0 // return an integer exit code
