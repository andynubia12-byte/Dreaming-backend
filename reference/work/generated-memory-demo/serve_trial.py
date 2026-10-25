from lettaproto.config import load_config
from lettaproto import db, ideastore, chat
from lettaproto.api import serve
config = load_config()
conn = db.connect(config.db_path)
db.init_db(conn)
ideastore.seed_prod(config, conn)
conn.close()
chat.TOOLS = [t for t in chat.TOOLS if t["function"]["name"] != "submit_memory_update"]
chat.AGENT_SYSTEM_ADDENDUM += "\nThis is a fixed-memory trial. Memory updates are disabled. Do not claim to update memory."
server = serve(config, port=8788, start_worker=False)
server.serve_forever()
