from pydantic import BaseModel

class ChessMoves(BaseModel):
	moves: str 