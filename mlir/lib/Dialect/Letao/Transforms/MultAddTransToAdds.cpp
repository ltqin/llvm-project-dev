//===- ConvertConst.cpp - Quantizes constant ops --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Letao/Passes.h"
#include "mlir/Dialect/Letao/LetaoOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/StandardTypes.h"
#include "llvm/Support/raw_ostream.h"


using namespace mlir;
using namespace mlir::letao;

namespace {
struct MultAddTransToAdds : public MultAddTransToAddsBase<MultAddTransToAdds> {
  void runOnFunction() override;
};

} // end anonymous namespace


void MultAddTransToAdds::runOnFunction() {
  FuncOp func = getFunction();

  func.walk([&](letao::MultiAddOp op) {
      //llvm::errs() << "Hello: ";
      //llvm::errs().write_escaped(op.getName()) << '\n';
  });
}

std::unique_ptr<OperationPass<FuncOp>> mlir::letao::createMultiAddTransPass() {
  return std::make_unique<MultAddTransToAdds>();
}
