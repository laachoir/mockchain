import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

class User:
	"""Capsules the transactions secret keys known to a certain user."""
	def __init__(self, name, usual_ringsize=1, transaction_frequency=0.4, mining_power=0, output_pick_strategy=None):
		self.name = name
		assert usual_ringsize > 0
		self.usual_ringsize = usual_ringsize
		self.transaction_frequency = transaction_frequency
		self.mining_power = mining_power
		self.output_pick_strategy = output_pick_strategy
		self.unspent_outputs = []

	def get_unspent_transaction(self):
		assert len(self.unspent_outputs) > 0
		# TODO create more realistic strategies
		if self.output_pick_strategy == "young_first":
			return sorted(self.unspent_outputs)[-1]
		if self.output_pick_strategy == "old_first":
			return sorted(self.unspent_outputs)[0]
		return np.random.choice(self.unspent_outputs)


class Mockchain:
	"""Create blocks and fill them with transactions"""
	def __init__(self, minimum_ringsize=1, confirmations_needed=1):
		assert minimum_ringsize > 0
		assert confirmations_needed > 0
		self.minimum_ringsize = minimum_ringsize
		self.confirmations_needed = confirmations_needed
		self.db = pd.DataFrame(columns=['block_height', 'transaction_hash', 'ring', 'real_input', 'sender', 'recipient'])
		self.graph = nx.DiGraph()
		self.current_block_height = 0
		self.current_transaction_num = 0
		self.num_transactions_in_confirmed_blocks = 0
		self.unconfirmed_blocks = []
		for _ in range(confirmations_needed):
			self.unconfirmed_blocks += [[]] # Caching seemingly prevents [[]] * confirmations_needed
		
	def mine_block(self, users):
		total_mining_power = sum([user.mining_power for user in users])
		miner = np.random.choice(users, p=[user.mining_power/total_mining_power for user in users])
		self.record_transaction(miner, miner, coinbase=True)
		eligible_users = [user for user in users if len(user.unspent_outputs) > 0]
		for user in eligible_users:
			if np.random.random() < user.transaction_frequency: #TODO no user behaviour change over time.
				transaction_recipient = np.random.choice(users) #TODO everybody has equal likelihood for now. Not realistic.
				self.record_transaction(user, transaction_recipient)
		for (recipient, transaction_hash) in self.unconfirmed_blocks[-1]:
			recipient.unspent_outputs += [transaction_hash]
			self.num_transactions_in_confirmed_blocks += 1
		self.unconfirmed_blocks = [[]] + self.unconfirmed_blocks[:-1]
		self.current_block_height += 1

	def record_transaction(self, transaction_author, transaction_recipient, coinbase=False):
		if coinbase:
			self.db.loc[len(self.db)] = [self.current_block_height, self.current_transaction_num, None, None, "__coinbase__", transaction_recipient.name]
			self.graph.add_edge(self.current_transaction_num, "__coinbase__", block_height=self.current_block_height, real_input=True, cluster="__coinbase__")
			self.unconfirmed_blocks[0] += [(transaction_recipient, self.current_transaction_num)]
			self.current_transaction_num += 1
			return True

		assert len(transaction_author.unspent_outputs) > 0
		desired_ring_size = transaction_author.usual_ringsize
		if self.minimum_ringsize > desired_ring_size:
			desired_ring_size = self.minimum_ringsize
		num_mixins = desired_ring_size - 1 # one for the real input
		# Check if there are enough mixins available to even create a big enough ring
		if desired_ring_size > self.num_transactions_in_confirmed_blocks:
			return False
		real_input = transaction_author.get_unspent_transaction()
		ring = [real_input]
		self.graph.add_edge(self.current_transaction_num, real_input, block_height=self.current_block_height, real_input=True, cluster=transaction_author.name)
		for _ in range(num_mixins):
			fake_input = real_input  # workaround to help checking for duplicate fake inputs.
			while fake_input in ring:
				fake_input = np.random.randint(self.num_transactions_in_confirmed_blocks) #TODO allow different distributions, especially gamma.
			ring += [fake_input]
			self.graph.add_edge(self.current_transaction_num, fake_input, block_height=self.current_block_height, real_input=False, cluster=transaction_author.name)
		ring = np.random.permutation(ring)
		self.db.loc[len(self.db)] = [self.current_block_height, self.current_transaction_num, ring, real_input, transaction_author.name, transaction_recipient.name]
		transaction_author.unspent_outputs.remove(real_input)
		self.unconfirmed_blocks[0] += [(transaction_recipient, self.current_transaction_num)]
		self.current_transaction_num += 1
		return True

if __name__ == '__main__':
	num_participants = 8
	num_total_blocks = 50

	usernames=["Alice", "Bob", "Carol", "Dave", "Erin", "Frank", "Geraldine", "Harold"]

	all_users = []
	chain = Mockchain(minimum_ringsize=5, confirmations_needed=5)
	for i in range(num_participants):
		# Only six usernames provided. Careful when setting a higher num_participants
		all_users += [User(usernames[i], mining_power=np.random.random(), transaction_frequency=0.2, output_pick_strategy="old_first")]
	stands_out = User("stands_out", mining_power=np.random.random(), transaction_frequency=0.2, usual_ringsize=50)
	all_users += [stands_out]
	for _ in range(num_total_blocks):
		chain.mine_block(all_users)
	chain.db.to_csv("gen_data.csv", sep='\t')
	nx.draw_networkx(chain.graph, pos=nx.drawing.layout.shell_layout(chain.graph), arrows=True, with_labels=True)
	plt.show()
	#plt.savefig("gen_data.png", bbox_inches='tight', dpi=300)
	# nx.write_graphml(chain.graph, "gen_data.graphml")
