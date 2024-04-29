// This creates an affine.apply
// run with the following to see the error: In the following pipeline remove the --lower-affine pass
// mlir-opt mlir/test/python/integration/dialects/diss_examples_expand_strided_metadata.mlir --split-input-file --convert-scf-to-cf -convert-func-to-llvm -expand-strided-metadata -finalize-memref-to-llvm --lower-affine --convert-arith-to-llvm -reconcile-unrealized-casts
func.func @subview_var_size(%0 : memref<64x64xf32, strided<[1, 1], offset: 0>>, %arg0 : index, %arg1 : index) -> memref<4x4xf32, strided<[1, 1], offset: ?>> {
  %chunk = memref.subview %0[%arg0, %arg0][4, 4][1, 1] :
    memref<64x64xf32, strided<[1, 1], offset: 0>>
    to memref<4x4xf32, strided<[1, 1], offset: ?>>
  %value = arith.constant 42.0 : f32
  scf.forall (%i, %j) = (0, 0) to (4, 4) step (1, 1){
    memref.store %value, %chunk[%i, %j] : memref<4x4xf32, strided<[1, 1], offset: ?>>
  }
  return %chunk : memref<4x4xf32, strided<[1, 1], offset: ?>>
}

// This does not emit the affine.apply in `expand_strided_metadata`
func.func @subview_const_size(%0 : memref<64x64xf32, strided<[1, 1], offset: 0>>) -> memref<4x4xf32, strided<[1, 1], offset: 0>> {
  %c4 = arith.constant 4 : index
  %chunk = memref.subview %0[0, 0][4, 4][1, 1] :
    memref<64x64xf32, strided<[1, 1], offset: 0>>
    to memref<4x4xf32, strided<[1, 1], offset: 0>>
  %value = arith.constant 42.0 : f32
  scf.forall (%i, %j) = (0, 0) to (4, 4) step (1, 1){
    memref.store %value, %chunk[%i, %j] : memref<4x4xf32, strided<[1, 1], offset: 0>>
  }
  return %chunk : memref<4x4xf32, strided<[1, 1], offset: 0>>
}

// This does not emit the affine.apply in `expand_strided_metadata`
func.func @subview_const_size0(%0 : memref<64x64xf32, strided<[1, 1], offset: 0>>) -> memref<4x4xf32, strided<[1, 1], offset: 0>> {
  %c4 = arith.constant 4 : index
  %chunk = memref.subview %0[0, 0][4, 4][1, 1] :
    memref<64x64xf32, strided<[1, 1], offset: 0>>
    to memref<4x4xf32, strided<[1, 1], offset: 0>>
  %value = arith.constant 42.0 : f32
  scf.forall (%i, %j) = (0, 0) to (4, 4) step (1, 1){
    memref.store %value, %chunk[%i, %j] : memref<4x4xf32, strided<[1, 1], offset: 0>>
  }
  return %chunk : memref<4x4xf32, strided<[1, 1], offset: 0>>
}

// This does not emit the affine.apply in `expand_strided_metadata`
func.func @subview_const_size1(%0 : memref<64x64xf32, strided<[1, 1], offset: 0>>) -> memref<4x4xf32, strided<[1, 1], offset: 0>> {
  %c4 = arith.constant 4 : index
  %chunk = memref.subview %0[0, 0][4, 4][1, 1] :
    memref<64x64xf32, strided<[1, 1], offset: 0>>
    to memref<4x4xf32, strided<[1, 1], offset: 0>>
  %value = arith.constant 42.0 : f32
  scf.forall (%i, %j) = (0, 0) to (4, 4) step (1, 1){
    memref.store %value, %chunk[%i, %j] : memref<4x4xf32, strided<[1, 1], offset: 0>>
  }
  return %chunk : memref<4x4xf32, strided<[1, 1], offset: 0>>
}

// This does not emit the affine.apply in `expand_strided_metadata`
func.func @subview_const_size2(%0 : memref<64x64xf32, strided<[1, 1], offset: 0>>) -> memref<4x4xf32, strided<[1, 1], offset: 0>> {
  %c4 = arith.constant 4 : index
  %chunk = memref.subview %0[0, 0][4, 4][1, 1] :
    memref<64x64xf32, strided<[1, 1], offset: 0>>
    to memref<4x4xf32, strided<[1, 1], offset: 0>>
  %value = arith.constant 42.0 : f32
  scf.forall (%i, %j) = (0, 0) to (4, 4) step (1, 1){
    memref.store %value, %chunk[%i, %j] : memref<4x4xf32, strided<[1, 1], offset: 0>>
  }
  return %chunk : memref<4x4xf32, strided<[1, 1], offset: 0>>
}

// This does not emit the affine.apply in `expand_strided_metadata`
func.func @subview_const_size3(%0 : memref<64x64xf32, strided<[1, 1], offset: 0>>) -> memref<4x4xf32, strided<[1, 1], offset: 0>> {
  %c4 = arith.constant 4 : index
  %chunk = memref.subview %0[0, 0][4, 4][1, 1] :
    memref<64x64xf32, strided<[1, 1], offset: 0>>
    to memref<4x4xf32, strided<[1, 1], offset: 0>>
  %value = arith.constant 42.0 : f32
  scf.forall (%i, %j) = (0, 0) to (4, 4) step (1, 1){
    memref.store %value, %chunk[%i, %j] : memref<4x4xf32, strided<[1, 1], offset: 0>>
  }
  return %chunk : memref<4x4xf32, strided<[1, 1], offset: 0>>
}

// This does not emit the affine.apply in `expand_strided_metadata`
func.func @subview_const_size4(%0 : memref<64x64xf32, strided<[1, 1], offset: 0>>) -> memref<4x4xf32, strided<[1, 1], offset: 0>> {
  %c4 = arith.constant 4 : index
  %chunk = memref.subview %0[0, 0][4, 4][1, 1] :
    memref<64x64xf32, strided<[1, 1], offset: 0>>
    to memref<4x4xf32, strided<[1, 1], offset: 0>>
  %value = arith.constant 42.0 : f32
  scf.forall (%i, %j) = (0, 0) to (4, 4) step (1, 1){
    memref.store %value, %chunk[%i, %j] : memref<4x4xf32, strided<[1, 1], offset: 0>>
  }
  return %chunk : memref<4x4xf32, strided<[1, 1], offset: 0>>
}

// This does not emit the affine.apply in `expand_strided_metadata`
func.func @subview_const_size5(%0 : memref<64x64xf32, strided<[1, 1], offset: 0>>) -> memref<4x4xf32, strided<[1, 1], offset: 0>> {
  %c4 = arith.constant 4 : index
  %chunk = memref.subview %0[0, 0][4, 4][1, 1] :
    memref<64x64xf32, strided<[1, 1], offset: 0>>
    to memref<4x4xf32, strided<[1, 1], offset: 0>>
  %value = arith.constant 42.0 : f32
  scf.forall (%i, %j) = (0, 0) to (4, 4) step (1, 1){
    memref.store %value, %chunk[%i, %j] : memref<4x4xf32, strided<[1, 1], offset: 0>>
  }
  return %chunk : memref<4x4xf32, strided<[1, 1], offset: 0>>
}

// This does not emit the affine.apply in `expand_strided_metadata`
func.func @subview_const_size6(%0 : memref<64x64xf32, strided<[1, 1], offset: 0>>) -> memref<4x4xf32, strided<[1, 1], offset: 0>> {
  %c4 = arith.constant 4 : index
  %chunk = memref.subview %0[0, 0][4, 4][1, 1] :
    memref<64x64xf32, strided<[1, 1], offset: 0>>
    to memref<4x4xf32, strided<[1, 1], offset: 0>>
  %value = arith.constant 42.0 : f32
  scf.forall (%i, %j) = (0, 0) to (4, 4) step (1, 1){
    memref.store %value, %chunk[%i, %j] : memref<4x4xf32, strided<[1, 1], offset: 0>>
  }
  return %chunk : memref<4x4xf32, strided<[1, 1], offset: 0>>
}