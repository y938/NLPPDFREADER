from tortoise import fields, models, Tortoise

class TextChunk(models.Model):
    id = fields.IntField(pk=True)
    content = fields.TextField()

class Embedding(models.Model):
    id = fields.IntField(pk=True)
    text_chunk = fields.ForeignKeyField('models.TextChunk', related_name='embeddings')
    vector = fields.JSONField()

Tortoise.init_models(["models"], "models")
