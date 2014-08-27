namespace Wavenet

module Main =

    open Wavelon
    open MathNet.Numerics.Data.Text

    [<EntryPoint>]
    let main argv = 
        let gazp = DelimitedReader.Read<float>("gazp.csv", false, ";", false, new System.Globalization.CultureInfo("en-US"))

        printfn "%s" (wtf.ToString())
        printfn "%A" argv
        0 // return an integer exit code
