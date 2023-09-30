/*
 * Copyright 2023 WebAssembly Community Group participants
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//
// Lowers Wasm GC to linear memory to polyfill for VMs that don't support it.
// The garbage collection and struct formats are left up to the user, as long
// as the data is present in linear memory with proper alignment.
//

#include "pass.h"
#include "wasm-builder.h"

namespace wasm {

static const Name ALLOCATE("__gc_lowering_allocate");
static const Name GET_RTT("__gc_lowering_get_rtt");
static const Name MEMSET8("__gc_lowering_memset8");
static const Name MEMSET16("__gc_lowering_memset16");
static const Name MEMSET32("__gc_lowering_memset32");
static const Name MEMSET64("__gc_lowering_memset64");
static const Name MEMSET128("__gc_lowering_memset128");

static const Name HELPER_NAMES[] = {
  ALLOCATE, GET_RTT, MEMSET8, MEMSET16, MEMSET32, MEMSET64};

static const Name GC_LOWERING_MEMORY("__gc_lowering_memory");

struct GCLowering
  : public WalkerPass<
      PostWalker<GCLowering, UnifiedExpressionVisitor<GCLowering>>> {

  void visitExpression(Expression* expr) {
    auto loweredType = lowerType(expr->type);
    if (expr->type != loweredType) {
      originalTypes[expr] = expr->type;
      expr->type = loweredType;
    }
  }

  void visitRefEq(RefEq* expr) {
    assert(expr->left->type == Type::i32 && expr->right->type == Type::i32);

    // Registering with originalTypes isn't necessary, since the type remains
    // the same.
    Builder builder(*getModule());
    replaceCurrent(
      builder.makeBinary(BinaryOp::EqInt32, expr->left, expr->right));
  }

  void visitRefNull(RefNull* expr) {
    if (!expr->type.isRef() ||
        expr->type.getHeapType().getBottom() != HeapType::none) {
      return;
    }

    Builder builder(*getModule());

    auto null = builder.makeConst<uint32_t>(0);
    originalTypes[null] = expr->type;
    replaceCurrent(null);
  }

  void visitI31New(I31New* expr) {
    Builder builder(*getModule());

    // This i31ref polyfill relies on GC allocations always having an alignment
    // of at least 2. That allows the LSB to signify whether a reference is an
    // i31ref.
    auto lowered = builder.makeBinary(
      BinaryOp::OrInt32,
      builder.makeBinary(
        BinaryOp::ShlInt32, expr->value, builder.makeConst<uint32_t>(1)),
      builder.makeConst<uint32_t>(1));

    originalTypes[lowered] = expr->type;
    replaceCurrent(lowered);
  }

  void visitI31Get(I31Get* expr) {
    Builder builder(*getModule());

    auto lowered = builder.makeBinary(expr->signed_ ? BinaryOp::ShrSInt32
                                                    : BinaryOp::ShrUInt32,
                                      expr->i31,
                                      builder.makeConst<uint32_t>(1));

    originalTypes[lowered] = expr->type;
    replaceCurrent(lowered);
  }

  void visitStructNew(StructNew* expr) {
    auto heapType = expr->type.getHeapType();
    auto& structInfo = getLoweredStructInfo(heapType);

    Builder builder(*getModule());

    auto structNewCall = builder.makeCall(
      getHelper(ALLOCATE),
      {builder.makeConst(structInfo.rttId), builder.makeConst(structInfo.size)},
      Type::i32);

    if (expr->isWithDefault()) {
      originalTypes[structNewCall] = expr->type;
      replaceCurrent(structNewCall);
      return;
    }

    auto structLocal = builder.addVar(getFunction(), Type::i32);
    auto block = builder.makeBlock(
      {builder.makeLocalSet(structLocal, structNewCall)}, Type::i32);

    for (size_t i = 0; i < structInfo.fields.size(); i++) {
      auto& field = structInfo.fields[i];
      block->list.push_back(
        builder.makeStore(field.size,
                          field.offset,
                          field.size,
                          builder.makeLocalGet(structLocal, Type::i32),
                          expr->operands[i],
                          field.loweredType,
                          memoryName));
    }

    block->list.push_back(builder.makeLocalGet(structLocal, Type::i32));
    originalTypes[block] = expr->type;
    replaceCurrent(block);
  }

  void visitStructGet(StructGet* expr) {
    lowerStructGetSet(
      expr,
      [&](Expression* structRef, Builder& builder, const FieldInfo& field) {
        return builder.makeLoad(field.size,
                                expr->signed_,
                                field.offset,
                                field.size,
                                structRef,
                                field.loweredType,
                                memoryName);
      });
  }

  void visitStructSet(StructSet* expr) {
    lowerStructGetSet(
      expr,
      [&](Expression* structRef, Builder& builder, const FieldInfo& field) {
        return builder.makeStore(field.size,
                                 field.offset,
                                 field.size,
                                 structRef,
                                 expr->value,
                                 field.loweredType,
                                 memoryName);
      });
  }

  void visitArrayNew(ArrayNew* expr) {
    auto heapType = expr->type.getHeapType();
    auto [loweredElemType, elemSize] =
      lowerFieldWithSize(heapType.getArray().element);

    Builder builder(*getModule());

    auto totalSize = builder.makeBinary(
      BinaryOp::MulInt32, expr->size, builder.makeConst(elemSize));

    if (expr->isWithDefault()) {
      auto allocation = builder.makeCall(
        getHelper(ALLOCATE),
        {builder.makeConst(getArrayRttId(heapType)), totalSize},
        Type::i32);
      originalTypes[allocation] = expr->type;
      replaceCurrent(allocation);
      return;
    }

    auto allocationLocal = builder.addVar(getFunction(), Type::i32);
    auto sizeLocal = builder.addVar(getFunction(), Type::i32);
    auto allocation =
      builder.makeCall(getHelper(ALLOCATE),
                       {builder.makeConst(getArrayRttId(heapType)),
                        builder.makeLocalTee(sizeLocal, totalSize, Type::i32)},
                       Type::i32);

    Name memsetTarget;
    if (elemSize == 1) {
      memsetTarget = getHelper(MEMSET8);
    } else if (elemSize == 2) {
      memsetTarget = getHelper(MEMSET16);
    } else if (elemSize == 4) {
      memsetTarget = getHelper(MEMSET32);
    } else if (elemSize == 8) {
      memsetTarget = getHelper(MEMSET64);
    } else if (elemSize == 16) {
      memsetTarget = getHelper(MEMSET128);
    } else {
      WASM_UNREACHABLE("unexpected element size for array");
    }

    auto memset = builder.makeCall(
      memsetTarget,
      {builder.makeLocalTee(allocationLocal, allocation, Type::i32),
       loweredElemType.isFloat()
         ? builder.makeUnary(elemSize == 4 ? UnaryOp::ReinterpretFloat32
                                           : UnaryOp::ReinterpretFloat64,
                             expr->init)
         : expr->init,
       builder.makeLocalGet(sizeLocal, Type::i32)},
      Type::none);

    auto block = builder.makeBlock(
      {memset, builder.makeLocalGet(allocationLocal, Type::i32)}, Type::i32);

    originalTypes[block] = expr->type;
    replaceCurrent(block);
  }

  void visitArrayNewFixed(ArrayNewFixed* expr) { WASM_UNREACHABLE("TODO"); }

  void visitArrayInitData(ArrayInitData* expr) { WASM_UNREACHABLE("TODO"); }

  void visitArrayInitElem(ArrayInitElem* expr) { WASM_UNREACHABLE("TODO"); }

  void visitFunction(Function* func) {
    auto signature = func->getSig();
    func->vars = lowerTypeList(func->vars);
    func->type =
      Signature(lowerType(signature.params), lowerType(signature.results));
  }

  void doWalkModule(Module* module) {
    auto& memories = module->memories;
    if (!memories.size()) {
      Builder builder(*module);
      module->addMemory(builder.makeMemory(GC_LOWERING_MEMORY));
      memoryName = GC_LOWERING_MEMORY;
    } else {
      memoryName = memories[0]->name;
    }

    structs.clear();
    arrays.clear();
    helpers.clear();

    for (auto helperName : HELPER_NAMES) {
      resolveHelper(helperName);
    }

    super::doWalkModule(module);
    originalTypes.clear();
  }

private:
  struct FieldInfo {
    Type loweredType;
    uint32_t size;
    uint32_t offset;
  };

  struct StructInfo {
    uint32_t rttId;
    uint32_t size;
    std::vector<FieldInfo> fields;
  };

  std::unordered_map<Name, Name> helpers;

  Name memoryName;
  std::unordered_map<HeapType, StructInfo> structs;
  std::unordered_map<HeapType, uint32_t> arrays;
  std::unordered_map<Expression*, Type> originalTypes;

  void resolveHelper(const Name& helperName) {
    // TODO: Use pass options to override the __gc_lowering_ prefixed names
    auto module = getModule();
    auto helperExport = module->getExportOrNull(helperName);

    if (!helperExport) {
      return;
    }

    if (helperExport->kind != ExternalKind::Function) {
      Fatal() << "GC helper " << helperName << " must be a function";
    }

    module->removeExport(helperName);
    helpers.insert({helperName, helperExport->value});
  }

  Name getHelper(const Name& helperName) {
    auto iterator = helpers.find(helperName);
    if (iterator == helpers.end()) {
      Fatal() << "missing GC helper: " << helperName;
    }

    return iterator->second;
  }

  const Type& getOriginalType(Expression* expr) {
    auto iterator = originalTypes.find(expr);
    return iterator != originalTypes.end() ? iterator->second : expr->type;
  }

  uint32_t nextRttId() {
    auto rttId = structs.size() + arrays.size();
    assert(rttId <= std::numeric_limits<uint32_t>::max());
    return static_cast<uint32_t>(rttId);
  }

  const StructInfo& getLoweredStructInfo(const HeapType& heapType) {
    auto iterator = structs.find(heapType);
    if (iterator != structs.end()) {
      return iterator->second;
    }

    std::vector<FieldInfo> fields;
    uint32_t offset = 0;

    for (auto& field : heapType.getStruct().fields) {
      auto [loweredType, size] = lowerFieldWithSize(field);

      // Align the field properly:
      auto mask = size - 1;
      if (offset & mask) {
        offset = (offset | mask) + 1;
      }

      fields.push_back({loweredType, size, offset});
      offset += size;
    }

    return structs[heapType] = {nextRttId(), offset, fields};
  }

  uint32_t getArrayRttId(const HeapType& heapType) {
    auto iterator = arrays.find(heapType);
    if (iterator != arrays.end()) {
      return iterator->second;
    }

    return arrays[heapType] = nextRttId();
  }

  Type lowerType(const Type& type) {
    if (type.isTuple()) {
      return lowerTypeList(type.getTuple());
    }

    if (!type.isRef()) {
      return type;
    }

    auto heapType = type.getHeapType();
    assert(heapType.getBottom() == HeapType::none && !heapType.isString() &&
           heapType != HeapType::stringview_iter &&
           heapType != HeapType::stringview_wtf16 &&
           heapType != HeapType::stringview_wtf8);
    return Type::i32;
  }

  TypeList lowerTypeList(const TypeList& types) {
    TypeList loweredTypes;
    for (auto& type : types) {
      loweredTypes.push_back(lowerType(type));
    }
    return loweredTypes;
  }

  std::pair<Type, uint32_t> lowerFieldWithSize(const Field& field) {
    auto loweredType = lowerType(field.type);
    auto size = field.type != loweredType ? loweredType.getByteSize()
                                          : field.getByteSize();
    return {loweredType, size};
  }

  template<typename StructGetSet, typename Callback>
  void lowerStructGetSet(StructGetSet* expr, Callback callback) {
    assert(expr->ref->type == Type::i32);
    auto& originalType = getOriginalType(expr->ref);
    auto& structInfo = getLoweredStructInfo(originalType.getHeapType());
    auto& field = structInfo.fields[expr->index];
    Builder builder(*getModule());

    Expression* lowered;
    if (originalType.isNullable()) {
      auto structLocal = builder.addVar(getFunction(), Type::i32);
      lowered = builder.makeIf(
        builder.makeLocalTee(structLocal, expr->ref, Type::i32),
        callback(builder.makeLocalGet(structLocal, Type::i32), builder, field),
        builder.makeUnreachable(),
        field.loweredType);
    } else {
      lowered = callback(expr->ref, builder, field);
    }

    originalTypes[lowered] = expr->type;
    replaceCurrent(lowered);
  }
};

Pass* createGCLoweringPass() { return new GCLowering(); }

} // namespace wasm