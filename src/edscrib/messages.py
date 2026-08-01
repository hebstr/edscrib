"""Every string this package paints on a page, and nothing else.

One home for the copy, because the same sentence rendered from two modules is two
sentences the day one of them is edited: a `MESSAGE_UNAVAILABLE` lived in two of them
under one name, in two languages, and which one an annotator met depended on which
subsystem had failed.

The language is the deployment's, French, and it is the annotator's rather than the
package's: what these say is read by a clinician in a browser. The module logs are the
operator's and stay in English, which is also where every detail goes, so nothing here
names a file, a column, a value or a path.

A message that prescribes a rebuild is addressed past the annotator, to whoever runs the
deployment, and says where the detail is so that the prescription can be applied at all.
It stops there: the annotator is told the state of the service and never how it broke.
"""

LABEL_USERNAME = "Identifiant"
LABEL_PASSWORD = "Mot de passe"
TITLE_LOGIN = "Connexion"

LABEL_TUTO = "**COMMENT UTILISER L'APPLICATION**"
LABEL_SAVE = "**ENREGISTRER ET PASSER AU SUIVANT**"
LABEL_DOWNLOAD = "**TÉLÉCHARGER LES DONNÉES ANNOTÉES**"

MESSAGE_REJECTED = "😕 Nom d'utilisateur ou mot de passe incorrect"

MESSAGE_UNAVAILABLE = (
    "Service indisponible : contactez l'administrateur. Le détail est dans le journal "
    "du serveur."
)

MESSAGE_MISMATCH = (
    "Les annotations ont été ouvertes sur d'autres documents : les identifiants ne "
    "correspondent plus, ni en valeur ni en ordre. Les données sont à reconstruire "
    "avant d'annoter. Le détail est dans le journal du serveur."
)

MESSAGE_LABELS = (
    "Les documents ne sont plus numérotés de façon continue, donc une position dans le "
    "parcours ne désigne plus le document affiché. Les données sont à reconstruire "
    "avant d'annoter. Le détail est dans le journal du serveur."
)

MESSAGE_DUPLICATES = (
    "Les documents ne sont plus identifiés un à un : un identifiant en désigne "
    "plusieurs, donc une sauvegarde ne peut plus dire sur lequel elle porte. Les "
    "données sont à reconstruire avant d'annoter. Le détail est dans le journal du "
    "serveur."
)

MESSAGE_COLUMNS = (
    "Les données ne portent plus toutes les colonnes que l'annotation déclare. Le "
    "détail est dans le journal du serveur."
)

MESSAGE_TYPES = (
    "Les données et les annotations ne lisent plus une même colonne comme un même "
    "type, donc leurs valeurs ne sont plus comparables. Les annotations déjà "
    "enregistrées sont intactes. Le détail est dans le journal du serveur."
)

MESSAGE_VALUES = (
    "Les annotations ont été ouvertes sur d'autres valeurs : une colonne des données "
    "ne correspond plus à ce que les annotations enregistrent. Le détail est dans le "
    "journal du serveur."
)

MESSAGE_CURSOR = (
    "L'annotation n'est pas configurée pour parcourir ses documents. Le détail est "
    "dans le journal du serveur."
)

MESSAGE_POSITION = (
    "Le parcours a changé depuis l'ouverture de cette page : la position affichée ne "
    "désigne plus de document. Rechargez la page pour repartir du dernier document "
    "annoté. Le détail est dans le journal du serveur."
)

MESSAGE_SAVE = (
    "L'annotation n'a pas été enregistrée, et le document sur lequel elle portait est "
    "inchangé. Rechargez la page avant de poursuivre. Le détail est dans le journal du "
    "serveur."
)
