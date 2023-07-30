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

#include "literal.h"
#include "pass.h"
#include "wasm-builder.h"
#include "wasm-traversal.h"
#include "wasm-type.h"
#include "wasm.h"
#include <limits>
#include <unordered_map>

namespace wasm {

static const Name STRUCT_NEW("__gc_lowering_struct_new");
static const Name GET_RTT("__gc_lowering_get_rtt");

struct GCLowering
  : public WalkerPass<
      PostWalker<GCLowering, UnifiedExpressionVisitor<GCLowering>>> {

  void visitExpression(Expression* expr) {
    if (expr->type.isStruct() || expr->type.isArray()) {
      originalTypes[expr] = expr->type;
      expr->type = Type::i32;
    }
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
    auto block =
      builder.makeBlock({builder.makeLocalSet(structLocal, structNewCall)});

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
    block->finalize(Type::i32, Block::NoBreak);
    originalTypes[block] = expr->type;
    replaceCurrent(block);
  }

  void visitStructGet(StructGet* expr) {
    auto& structInfo =
      getLoweredStructInfo(getOriginalType(expr->ref).getHeapType());
    auto& field = structInfo.fields[expr->index];
    Builder builder(*getModule());

    auto load = builder.makeLoad(field.size,
                                 expr->signed_,
                                 field.offset,
                                 field.size,
                                 expr->ref,
                                 field.loweredType,
                                 memoryName);
    load->finalize();
    originalTypes[load] = expr->type;
    replaceCurrent(load);
  }

  void doWalkModule(Module* module) {
    memoryName = module->memories[0]->name;
    structs.clear();
    arrays.clear();
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

  Name memoryName;
  std::unordered_map<HeapType, StructInfo> structs;
  std::unordered_map<HeapType, uint32_t> arrays;
  // FIXME: The pass probably shouldn't be storing every replaced node in a map.
  std::unordered_map<Expression*, Type> originalTypes;

  // FIXME: See above.
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
    auto& loweredType = type.isStruct() || type.isArray() ? Type::i32 : type;
    assert(!loweredType.isRef());
    return loweredType;
  }
};

Pass* createGCLoweringPass() { return new GCLowering(); }

} // namespace wasm