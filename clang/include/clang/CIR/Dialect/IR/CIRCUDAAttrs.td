//===---- CIRCUDAAttrs.td - CIR dialect attrs for CUDA -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the CIR dialect attributes for CUDA.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_DIALECT_IR_CIRCUDAATTRS_TD
#define CLANG_CIR_DIALECT_IR_CIRCUDAATTRS_TD

//===----------------------------------------------------------------------===//
// CUDAKernelNameAttr
//===----------------------------------------------------------------------===//

def CIR_CUDAKernelNameAttr : CIR_Attr<"CUDAKernelName", "cu.kernel_name"> {
  let summary = "Device-side kernel name.";
  let description =
  [{
    This attribute records the device-side kernel name on a stub definition
    or, for HIP RDC, on the host kernel handle.

    The host symbol can differ from this name, including when an internal
    device kernel is externalized.
  }];

  let parameters = (ins "mlir::StringAttr":$kernel_name);
  let assemblyFormat = "`<` $kernel_name `>`";
  let canHaveIllegalCXXABIType = 0;
}

def CIR_CUDAExternallyInitializedAttr : CIR_Attr<"CUDAExternallyInitialized",
                                             "cu.externally_initialized"> {
  let summary = "The marked variable is externally initialized.";
  let description =
  [{
    CUDA __device__ and __constant__ variables, along with surface and
    textures, might be initialized by host, hence "externally initialized".
    Therefore they must be emitted even if they are not referenced.

    The attribute corresponds to the attribute on LLVM with the same name.
  }];
  let canHaveIllegalCXXABIType = 0;
}
// No wrapper attribute: the kind is only ever printed by
// CIR_CUDAVarRegistrationInfoAttr's own assembly format.
def CIR_CUDADeviceVarKind : CIR_I32Enum<"CUDADeviceVarKind",
    "CUDA device variable kind", [
  I32EnumCase<"Variable", 0>,
  I32EnumCase<"Surface", 1>,
  I32EnumCase<"Texture", 2>,
]>;

// Attribute carrying device variable registration flags
def CIR_CUDAVarRegistrationInfoAttr : CIR_Attr<"CUDAVarRegistrationInfo", "cu.var_registration"> {
  let summary = "Device variable registration flags.";
  let parameters = (ins
    "std::string":$device_side_name,
    "CUDADeviceVarKind":$kind,
    "bool":$isExtern,
    "bool":$isConstant,
    "bool":$isManaged,
    "bool":$isNormalized,
    "int32_t":$surfTexType
  );

  let hasCustomAssemblyFormat = 1;
}

#endif // CLANG_CIR_DIALECT_IR_CIRCUDAATTRS_TD
