module {
  func.func @main(
      %activation: memref<4x64xf16>,
      %weight: memref<64x64xf16>,
      %output: memref<4x64xf16>) {
    %zero = arith.constant 0.000000e+00 : f16
    linalg.fill ins(%zero : f16) outs(%output : memref<4x64xf16>)
    linalg.matmul
        ins(%activation, %weight : memref<4x64xf16>, memref<64x64xf16>)
        outs(%output : memref<4x64xf16>)
    return
  }
}
