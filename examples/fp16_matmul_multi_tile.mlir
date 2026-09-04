module {
  func.func @main(
      %activation: memref<40x33xf16>,
      %weight: memref<33x45xf16>,
      %output: memref<40x45xf16>) {
    %zero = arith.constant 0.000000e+00 : f16
    linalg.fill ins(%zero : f16) outs(%output : memref<40x45xf16>)
    linalg.matmul
        ins(%activation, %weight : memref<40x33xf16>, memref<33x45xf16>)
        outs(%output : memref<40x45xf16>)
    return
  }
}
