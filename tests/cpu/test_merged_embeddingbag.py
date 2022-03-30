import torch
import torch.nn as nn
import unittest
import copy
from torch.testing._internal.common_utils import TestCase
from intel_extension_for_pytorch.nn.modules import MergedEmbeddingBagWithSGD as MergedEmbeddingBagWithSGD

class TestMergedEmbeddingBagWithSGD(TestCase):

    table0 = nn.EmbeddingBag(100, 16, mode='mean').double()
    table1 = nn.EmbeddingBag(50, 32, mode='sum')
    table2 = nn.EmbeddingBag(18000000, 8, mode='sum', include_last_offset=True, _weight=torch.empty(18000000, 8, dtype=torch.bfloat16))
    merged = MergedEmbeddingBagWithSGD.from_embeddingbag_list([table0, table1, table2])
    merged2 = MergedEmbeddingBagWithSGD([
        (100, 16, 'mean', table0.weight.dtype, table0.weight.detach()),
        (50, 32, 'sum', table1.weight.dtype, table1.weight.detach()),
        (18000000, 8, 'sum', table2.weight.dtype, table2.weight.detach()),
    ])

    input = [
        [torch.LongTensor([10, 10, 15, 10, 20, 25]), torch.LongTensor([[0, 30], [21, 15], [30, 11]]), torch.LongTensor([10, 15, 17999999])],
        [torch.LongTensor([0, 1, 3]), None, torch.LongTensor([0, 1, 2, 3])],
        [table0.include_last_offset, table1.include_last_offset, table2.include_last_offset]
    ]

    expected_input = (
        torch.LongTensor([10, 10, 15, 10, 20, 25, 0, 30, 21, 15, 30, 11, 10, 15, 17999999]),
        torch.LongTensor([0, 1, 3, 6, 8, 10, 12, 13, 14, 15]),
        torch.LongTensor([10, 10, 15, 10, 20, 25, 100, 130, 121, 115, 130, 111, 160, 165, 18000149])
    )

    expected_indices_weight_for_update = {
        10: 1 + 1 / 2 + 1 / 3,
        15: 1 / 2, 20: 1 / 3, 25: 1 / 3,
        100: 1, 111: 1, 115: 1, 121: 1, 130: 2,
        160: 1, 165: 1, 18000149: 1
    }

    def test_create_from_embedingbaglist_vs_create_from_init_function(self):
        self.assertEqual(self.merged.weights, self.merged2.weights)
        self.assertEqual(self.merged.pooling_modes, self.merged2.pooling_modes)
        self.assertEqual(self.merged.dtypes, self.merged2.dtypes)

    def test_input_prepare_function(self):
        merged_indices, merged_offsets, merged_indices_with_row_offsets = self.merged.linearize_indices_and_offsets(*self.input)
        self.assertEqual(self.merged.linearize_indices_and_offsets(*self.input), self.expected_input)
        self.assertEqual(self.merged(self.expected_input, torch.BoolTensor([False])), self.merged2(self.input))

    def _test_inference_only(self, model):
        with torch.no_grad():
            outputs = model(self.expected_input, torch.BoolTensor([False]))
            ref_out0 = self.table0(self.input[0][0], self.input[1][0])
            ref_out1 = self.table1(self.input[0][1], self.input[1][1])
            ref_out2 = self.table2(self.input[0][2], self.input[1][2])
            self.assertEqual(outputs[0], ref_out0)
            self.assertEqual(outputs[1], ref_out1)
            self.assertEqual(outputs[2], ref_out2)

    @unittest.skipIf(True, "Pytorch has trace issue for Modules with ParameterList")
    def test_inference(self):
        model = copy.deepcopy(self.merged)
        self._test_inference_only(model)
        with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
            self._test_inference_only(model)
        with torch.no_grad():
            trace_model = torch.jit.trace(model, [self.expected_input, torch.BoolTensor([False])])
        self._test_inference_only(trace_model)

    def get_local_indice(self, indice):
        table_id = 0
        while (indice >= self.merged.row_offsets[table_id + 1]):
            table_id += 1
        logical_indice = indice - self.merged.row_offsets[table_id].item()
        return table_id, logical_indice

    def test_training(self):
        model = copy.deepcopy(self.merged)
        outputs = model(self.expected_input, torch.BoolTensor([False]))
        loss = outputs[0].sum() + outputs[1].sum() + outputs[2].sum()
        default_lr = 0.01
        weights = copy.deepcopy(model.weights)
        loss.backward()
        updated_weights = model.weights
        for indice in self.expected_indices_weight_for_update:
            table_id, logical_indice = self.get_local_indice(indice)
            # grad will be all "1" for "sum"
            grad = torch.ones(1, dtype=weights[table_id].dtype) * self.expected_indices_weight_for_update[indice]
            ref_updated_weight = weights[table_id][logical_indice] - default_lr * grad
            self.assertEqual(updated_weights[table_id][logical_indice], ref_updated_weight)

    def test_training_with_weight_decay(self):
        import bench.custom_op_bench.optimizer
        sgd = bench.custom_op_bench.optimizer.non_fused_sgd
        model = MergedEmbeddingBagWithSGD.from_embeddingbag_list(
            [self.table0, self.table1, self.table2],
            lr=1.0,
            weight_decay=0.1
        )
        outputs = model(self.expected_input, torch.BoolTensor([False]))
        sgd_args = copy.deepcopy(model.sgd_args)
        loss = outputs[0].sum() + outputs[1].sum() + outputs[2].sum()
        weights = copy.deepcopy(model.weights)
        loss.backward()
        updated_weights = model.weights
        for indice in self.expected_indices_weight_for_update:
            table_id, logical_indice = self.get_local_indice(indice)
            # grad will be all "1" for "sum"
            ref_updated_weight = weights[table_id][logical_indice].detach().clone()
            grad = torch.ones_like(ref_updated_weight, dtype=weights[table_id].dtype) * self.expected_indices_weight_for_update[indice]
            sgd(
                ref_updated_weight,
                grad,
                None,
                0,  # momentum
                sgd_args.lr,
                sgd_args.weight_decay,
                0,  # dampening
                False  # nestrov
            )
            self.assertEqual(updated_weights[table_id][logical_indice], ref_updated_weight, rtol=0.01, atol=0.01)

    def test_cast_bfloat16(self):
        model = copy.deepcopy(self.merged)
        model.to_bfloat16_train()
        w0 = torch.ops.torch_ipex.cat_bfloat16_float(model.weights[0], model.sgd_args.bf16_trail[0])
        self.assertEqual(w0, self.table0.weight.float())

        w1 = torch.ops.torch_ipex.cat_bfloat16_float(model.weights[1], model.sgd_args.bf16_trail[1])
        self.assertEqual(w1, self.table1.weight)

        w2 = model.weights[2]
        self.assertEqual(w2, self.table2.weight)
        self.assertEqual(torch.zeros_like(w2, dtype=torch.bfloat16), model.sgd_args.bf16_trail[2])




if __name__ == '__main__':
    test = unittest.main()