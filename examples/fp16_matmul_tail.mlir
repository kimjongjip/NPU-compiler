module {
  func.func @main(
      %activation: memref<5x96xf16>,
      %weight: memref<96x37xf16>,
      %output: memref<5x37xf16>) {
    %zero = arith.constant 0.000000e+00 : f16
    linalg.fill ins(%zero : f16) outs(%output : memref<5x37xf16>)
    linalg.matmul
        ins(%activation, %weight : memref<5x96xf16>, memref<96x37xf16>)
        outs(%output : memref<5x37xf16>)
    return
  }
}
