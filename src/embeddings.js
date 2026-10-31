// Embeds support tickets so similar tickets can be found with a vector index.
// The index column is sized from EMBEDDING_DIMENSIONS, so it must match the model.
export const EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v3';
export const EMBEDDING_DIMENSIONS = 512;

export async function embedTickets(tickets, embed) {
  const vectors = await embed(EMBEDDING_MODEL, tickets.map((ticket) => ticket.body));
  return tickets.map((ticket, index) => {
    if (vectors[index].length !== EMBEDDING_DIMENSIONS)
      throw new Error(`Unexpected embedding size for ticket ${ticket.id}`);
    return { id: ticket.id, vector: vectors[index] };
  });
}
