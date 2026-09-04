module {
  func.func @main(
      %activation: memref<4x64xf16>,
      %weight: memref<64x64xf16>,
      %output: memref<4x64xf16>) {
    linalg.matmul
        ins(%activation, %weight : memref<4x64xf16>, memref<64x64xf16>)
        outs(%output : memref<4x64xf16>)
    return
  }
}
