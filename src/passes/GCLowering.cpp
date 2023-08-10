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
#include "wasm.h"

namespace wasm {

static const Name STRUCT_NEW("__gc_lowering_struct_new");
static const Name GET_RTT("__gc_lowering_get_rtt");

static const std::unordered_map<Name, Name> DUMMY_EXPORTS = {
  {"__gc_lowering_struct_new_dummy_export", STRUCT_NEW},
  {"__gc_lowering_get_rtt_dummy_export", GET_RTT}};

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

    auto null = builder.makeConst(Literal::makeZero(Type::i32));
    originalTypes[null] = expr->type;
    replaceCurrent(null);
  }

  void visitStructNew(StructNew* expr) {
    auto heapType = expr->type.getHeapType();
    auto& structInfo = getLoweredStructInfo(heapType);

    Builder builder(*getModule());

    auto structNewCall = builder.makeCall(
      STRUCT_NEW,
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

    super::doWalkModule(module);
    module->removeExports([&](Export* exportMember) {
      if (exportMember->kind != ExternalKind::Function) {
        return false;
      }

      auto iterator = DUMMY_EXPORTS.find(exportMember->name);
      if (iterator == DUMMY_EXPORTS.end()) {
        return false;
      }

      return iterator->second == exportMember->value;
    });

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

  Name memoryName;
  std::unordered_map<HeapType, StructInfo> structs;
  std::unordered_map<HeapType, uint32_t> arrays;
  std::unordered_map<Expression*, Type> originalTypes;

  const Type& getOriginalType(Expression* expr) {
    auto iterator = originalTypes.find(expr);
    return iterator != originalTypes.end() ? iterator->second : expr->type;
  }

  const StructInfo& getLoweredStructInfo(const HeapType& heapType) {
    auto iterator = structs.find(heapType);
    if (iterator != structs.end()) {
      return iterator->second;
    }

    std::vector<FieldInfo> fields;
    uint32_t offset = 0;

    for (auto& field : heapType.getStruct().fields) {
      auto loweredType = lowerType(field.type);
      auto size =
        field.type.isStruct() || field.type.isArray() ? 4 : field.getByteSize();

      // Align the field properly:
      auto mask = size - 1;
      if (offset & mask) {
        offset = (offset | mask) + 1;
      }

      fields.push_back({loweredType, size, offset});
      offset += size;
    }

    auto rttId = structs.size() + arrays.size();
    assert(rttId <= std::numeric_limits<uint32_t>::max());
    return structs[heapType] = {static_cast<uint32_t>(rttId), offset, fields};
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