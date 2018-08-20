import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

class User:
	"""Capsules the transactions secret keys known to a certain user."""
	def __init__(self, name, usual_ringsize=1, transaction_frequency=0.4, mining_power=0):
		self.name = name
		assert usual_ringsize > 0
		self.usual_ringsize = usual_ringsize
		self.transaction_frequency = transaction_frequency
		self.mining_power = mining_power
		self.unspent_outputs = []


class Mockchain:
	"""Create blocks and fill them with transactions"""
	def __init__(self, minimum_ringsize):
		assert minimum_ringsize > 0
		self.minimum_ringsize = minimum_ringsize
		self.db = pd.DataFrame(columns=['block_height', 'transaction_hash', 'ring', 'real_input', 'sender', 'recipient'])
		self.current_block_height = 0
		self.current_transaction_num = 0
		self.num_transactions_in_blocks = 0
		
	def mine_block(self, users):
		transactions_in_block = 0
		total_mining_power = sum([user.mining_power for user in users])
		miner = np.random.choice(users, p=[user.mining_power/total_mining_power for user in users])
		transactions_in_block += self.record_transaction(miner, miner, coinbase=True)
		eligible_users = [user for user in users if len(user.unspent_outputs) > 0]
		for user in eligible_users:
			if np.random.random() < user.transaction_frequency: #TODO no user behaviour change over time.
				transaction_recipient = np.random.choice(users) #TODO everybody has equal likelihood for now. Not realistic.
				transactions_in_block += self.record_transaction(user, transaction_recipient)
		self.num_transactions_in_blocks += transactions_in_block
		self.current_block_height += 1

	def record_transaction(self, transaction_author, transaction_recipient, coinbase=False):
		if coinbase:
			self.db.loc[len(self.db)] = [self.current_block_height, self.current_transaction_num, None, None, "__coinbase__", transaction_recipient.name]
			transaction_recipient.unspent_outputs += [self.current_transaction_num]
			self.current_transaction_num += 1
			return True

		assert len(transaction_author.unspent_outputs) > 0
		desired_ring_size = transaction_author.usual_ringsize
		if self.minimum_ringsize > desired_ring_size:
			desired_ring_size = self.minimum_ringsize
		num_mixins = desired_ring_size - 1 # one for the real input
		# Check if there are enough mixins available to even create a big enough ring
		if desired_ring_size > self.num_transactions_in_blocks:
			return False
		real_input = np.random.choice(transaction_author.unspent_outputs)
		ring = [real_input]
		for _ in range(num_mixins):
			fake_input = real_input  # workaround to help checking for duplicate fake inputs.
			while fake_input in ring:
				fake_input = np.random.randint(self.num_transactions_in_blocks) #TODO allow different distributions, especially gamma.
				print(num_mixins, self.num_transactions_in_blocks, ring, fake_input)
			ring += [fake_input]
		ring = np.random.permutation(ring)
		self.db.loc[len(self.db)] = [self.current_block_height, self.current_transaction_num, ring, real_input, transaction_author.name, transaction_recipient.name]
		transaction_author.unspent_outputs.remove(real_input)
		transaction_recipient.unspent_outputs += [self.current_transaction_num]
		self.current_transaction_num += 1
		return True

if __name__ == '__main__':
	num_participants = 4
	num_total_blocks = 20

	all_users = []
	chain = Mockchain(4)
	for i in range(num_participants):
		all_users += [User(i, mining_power=i+1, transaction_frequency=0.8)]
	for _ in range(num_total_blocks):
		chain.mine_block(all_users)
	print(chain.db)