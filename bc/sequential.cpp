#include "sequential.h"

std::vector<float> bc_cpu(graph g, const std::set<int> &source_vertices)
{
	std::vector<float> bc(g.n,0);
	int end = source_vertices.empty() ? g.n : source_vertices.size();
	std::set<int>::iterator it = source_vertices.begin();

	for(int k=0; k<end; k++)
	{
		int i = source_vertices.empty() ? k : *it++;
		std::queue<int> Q;
		std::stack<int> S;
		std::vector<int> d(g.n,INT_MAX);
		d[i] = 0;
		std::vector<unsigned long long> sigma(g.n,0);
		sigma[i] = 1;
		std::vector<float> delta(g.n,0);
		Q.push(i);

		while(!Q.empty())
		{
			int v = Q.front();
			Q.pop();
			S.push(v);
			for(int j=g.R[v]; j<g.R[v+1]; j++)
			{
				int w = g.C[j];
				if(d[w] == INT_MAX)
				{
					Q.push(w);
					d[w] = d[v]+1;
				}
				if(d[w] == (d[v]+1))
				{
					sigma[w] += sigma[v];
				}
			}
		}

		while(!S.empty())
		{
			int w = S.top();
			S.pop();
			for(int j=g.R[w]; j<g.R[w+1]; j++)
			{
				int v = g.C[j];
				if(d[v] == (d[w] - 1))
				{
					delta[v] += (sigma[v]/(float)sigma[w])*(1+delta[w]);
				}
			}	
			
			if(w != i)
			{
				bc[w] += delta[w];
			}
		}
	}

	for(int i=0; i<g.n; i++)
	{
		bc[i] /= 2.0f; //Undirected edges are modeled as two directed edges, but the scores shouldn't be double counted.
	}

	return bc;
}
